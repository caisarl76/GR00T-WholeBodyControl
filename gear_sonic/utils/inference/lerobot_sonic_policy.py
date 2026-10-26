"""Client and 50 Hz output for LeRobot SONIC-token policies (e.g. the GR00T handover models).

LeRobot policies are served by lerobot's ``examples/g1_dex3_training/sonic_policy_server.py``, not by an
Isaac-GR00T PolicyServer (ZMQ REQ/REP, msgpack):

  {"op": "info"}  -> {"image_keys": [...], "n_obs": int, "shapes": {key: [C, H, W]}, "model": path}
  {"op": "chunk", "states": [[28 floats]], "images": [{key: jpeg}], "task": str}
                  -> {"chunk": float32 bytes, "shape": [T, 78], "latency_s": float}

The client sends what lerobot's ``sonic_policy_streamer.py`` sends:
- the 28D state: arms (motors 15-28), then the left and right Dex3 hands, the right hand in dataset order;
- the head image as an RGB JPEG at quality 90;
- the task.

It returns each chunk split the way the native loop expects: motion token 64, left hand 7, right hand 7.

``TokenStream`` reproduces the streamer's 50 Hz output (``sonic_token_stream.ChunkResampler`` and
``SonicBackend.send``):
- each chunk plays from its observation time, interpolated at the policy rate;
- a new chunk cross-fades from the current output over ``blend_s``;
- the token is slew-limited to ``max_token_step`` per tick.
"""

import time

import numpy as np
import zmq

# Dataset right-hand order is thumb 0-2, index 0-1, middle 0-1. NVIDIA's MuJoCo bridge orders the right Dex3 slots
# thumb, middle, index; "swap" converts both ways (an involution), as the streamer's --dex3-right-order swap.
RIGHT_SWAP = np.array([0, 1, 2, 5, 6, 3, 4])


def lerobot_state(state_msg, right_order):
    """28D policy state from a g1_debug message: body_q[15:29], left Dex3, right Dex3 (dataset order)."""
    body = np.asarray(state_msg["body_q"], np.float32).reshape(-1)
    if body.shape != (29,):
        raise ValueError(f"expected 29 body joints, got {body.shape}")
    left = np.asarray(state_msg["left_hand_q"], np.float32).reshape(-1)
    right = np.asarray(state_msg["right_hand_q"], np.float32).reshape(-1)
    if left.shape != (7,) or right.shape != (7,):
        raise ValueError("expected 7 joints per Dex3 hand")
    return np.concatenate([body[15:29], left, right[right_order]]).astype(np.float32)


class LeRobotSonicPolicyClient:
    """Drop-in for the Isaac-GR00T PolicyClient in the native loop, for ``sonic_policy_server.py``.

    ``get_action`` takes ``observation["lerobot"] = {"state": 28 floats, "image": HxWx3 RGB, "task": str}``.
    A request that times out drops the socket and raises (lazy pirate), like the streamer's RemotePolicy.
    """

    def __init__(self, host, port, timeout_ms, right_order="dataset"):
        if right_order not in ("dataset", "swap"):
            raise ValueError("right_order must be 'dataset' or 'swap'")
        self.addr, self.timeout_ms, self.sock = f"tcp://{host}:{port}", int(timeout_ms), None
        self.right_order = RIGHT_SWAP if right_order == "swap" else np.arange(7)
        info = self._ask({"op": "info"})
        if int(info["n_obs"]) != 1 or len(info["image_keys"]) != 1:
            raise ValueError(f"only single-observation, single-camera policies are supported: {info}")
        self.image_key = info["image_keys"][0]
        self.image_shape = tuple(info["shapes"][self.image_key])  # (C, H, W)
        self.model = info.get("model")

    def _ask(self, request):
        import msgpack

        if self.sock is None:
            self.sock = zmq.Context.instance().socket(zmq.REQ)
            self.sock.setsockopt(zmq.LINGER, 0)
            self.sock.connect(self.addr)
        self.sock.send(msgpack.packb(request, use_bin_type=True))
        if not self.sock.poll(self.timeout_ms):
            self.close()
            raise TimeoutError(f"policy server {self.addr} did not answer within {self.timeout_ms} ms")
        reply = msgpack.unpackb(self.sock.recv(), raw=False)
        if "error" in reply:
            raise RuntimeError(f"policy server: {reply['error']}")
        return reply

    def ping(self):
        try:
            self._ask({"op": "info"})
            return True
        except (TimeoutError, RuntimeError, zmq.ZMQError):
            return False

    def get_action(self, observation):
        import cv2

        inputs = observation["lerobot"]
        state = np.asarray(inputs["state"], np.float32).reshape(-1)
        image = np.asarray(inputs["image"])
        if state.shape != (28,) or not np.isfinite(state).all():
            raise ValueError("expected a finite 28D policy state")
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"expected an HxWx3 uint8 RGB image, got {image.shape} {image.dtype}")
        # RGB in, RGB out: neither side converts colours (the server resizes to its input shape).
        jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        reply = self._ask(
            {"op": "chunk", "states": [state.tolist()], "images": [{self.image_key: jpeg}], "task": inputs["task"]}
        )
        chunk = np.frombuffer(reply["chunk"], np.float32).reshape(reply["shape"])
        if chunk.ndim != 2 or chunk.shape[1] != 78 or not np.isfinite(chunk).all():
            raise ValueError(f"expected a finite [T, 78] chunk, got {chunk.shape}")
        action = {
            "motion_token": chunk[None, :, :64],
            "left_hand_joints": chunk[None, :, 64:71],
            "right_hand_joints": chunk[None, :, 71:78][..., self.right_order],  # Deploy slot order.
        }
        return action, {"latency_s": reply.get("latency_s"), "model": self.model}

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None


class TokenStream:
    """The streamer's 50 Hz token and hand output from timed chunks.

    ``set_chunk(rows, t0, now)``: row i of ``rows`` ([T, 78]: token 64, hands 14) is the target for time
    ``t0 + i / source_fps``. Before t0 the first row is returned; after the last row it is held. With
    ``blend_s > 0``, a new chunk cross-fades linearly from what is being output at ``now``. ``at(t)`` then
    slew-limits the token to ``max_token_step`` per call (hands are not limited), starting from
    ``start_token`` after ``reset``. With ``anchor_s > 0``, a new chunk's token dims instead continue from
    ``last_token`` (the token actually sent) and the offset decays with time constant ``anchor_s``; the hands
    keep the cross-fade.
    """

    def __init__(self, source_fps, blend_s=0.0, max_token_step=0.0, anchor_s=0.0):
        if source_fps <= 0 or blend_s < 0 or max_token_step < 0 or anchor_s < 0:
            raise ValueError("source_fps must be positive; blend_s, max_token_step, anchor_s non-negative")
        self.source_fps, self.blend_s, self.max_token_step = float(source_fps), float(blend_s), float(max_token_step)
        self.anchor_s = float(anchor_s)
        self.reset()

    def reset(self, start_token=None):
        self._chunk, self._t0, self._prev, self._blend, self._anchor = None, 0.0, None, (0.0, 0.0), None
        self.last_token = None if start_token is None else np.asarray(start_token, np.float32).reshape(64).copy()

    @property
    def ready(self):
        return self._chunk is not None

    def set_chunk(self, rows, t0, now):
        rows = np.asarray(rows, np.float32)
        if rows.ndim != 2 or rows.shape[1] != 78 or len(rows) == 0 or not np.isfinite(rows).all():
            raise ValueError(f"expected a nonempty finite [T, 78] chunk, got {rows.shape}")
        if self.anchor_s > 0 and self._chunk is not None:
            # Continue from the token being sent (itself possibly still offset) with the new chunk's motion.
            ref = self._row_at(now)[:64] if self.last_token is None else self.last_token
            self._anchor = (ref - self._interpolate(rows, float(t0), now)[:64], float(now))
        else:
            self._anchor = None
        if self.blend_s > 0 and self._chunk is not None:
            # Fade from what is being output now (itself possibly mid-fade) to the new chunk.
            self._prev = (self._row_at(now)[None], now) if self._prev is not None else (self._chunk, self._t0)
            self._blend = (float(now), self.blend_s)
        else:
            self._prev = None
        self._chunk, self._t0 = rows, float(t0)

    def _interpolate(self, rows, t0, t):
        x = min(max((t - t0) * self.source_fps, 0.0), len(rows) - 1)
        i = int(np.floor(x + 1e-9))
        w = x - i
        if w < 1e-9 or i + 1 >= len(rows):
            return rows[i].copy()
        return ((1 - w) * rows[i] + w * rows[i + 1]).astype(np.float32)

    def _row_at(self, t):
        new = out = self._interpolate(self._chunk, self._t0, t)
        if self._prev is not None:
            start, duration = self._blend
            w = (t - start) / duration
            if w < 1:
                old = self._interpolate(*self._prev, t)
                w = max(w, 0.0)
                out = ((1 - w) * old + w * new).astype(np.float32)
            else:
                self._prev = None
        if self._anchor is not None:
            offset, start = self._anchor
            out = out.copy()
            out[:64] = new[:64] + offset * np.float32(np.exp(-max(t - start, 0.0) / self.anchor_s))
        return out

    def at(self, t):
        """(token 64, hands 14) to send at time t."""
        if self._chunk is None:
            raise RuntimeError("no chunk set")
        row = self._row_at(t)
        token = row[:64]
        if self.max_token_step > 0 and self.last_token is not None:
            token = self.last_token + np.clip(token - self.last_token, -self.max_token_step, self.max_token_step)
        self.last_token = token.astype(np.float32)
        return self.last_token.copy(), row[64:].copy()


if __name__ == "__main__":
    stream = TokenStream(50, blend_s=0.3, max_token_step=0.05)
    rows = np.zeros((40, 78), np.float32)
    rows[:, :64] = np.linspace(0, 1, 40)[:, None]
    stream.reset(start_token=np.zeros(64))
    stream.set_chunk(rows, t0=0.0, now=0.0)
    token, hands = stream.at(0.5)
    assert np.allclose(token, 0.05) and hands.shape == (14,)  # Slew-limited from the start token.
    print("ok")

// Test-only extension: exercises the production callback/getters without the SDK/service.
#include "py_bindings.cpp"
#include <cstring>

extern "C" {
int PXREAInit(void*, pfPXREAClientCallback, unsigned) { return -1; }
int PXREADeinit() { return -1; }
int PXREADeviceControlJson(const char*, const char*) { return -1; }
int PXREASendBytesToDevice(const char*, const char*, unsigned) { return -1; }
}

void injectPacket(const std::string& value) {
    const auto packet = json{{"value", value}}.dump();
    PXREADevStateJson state{};
    if (packet.size() >= sizeof(state.stateJson)) throw std::runtime_error("fixture packet too large");
    std::memcpy(state.stateJson, packet.c_str(), packet.size() + 1);
    OnPXREAClientCallback(nullptr, PXREADeviceStateJson, 0, &state);
}

PYBIND11_MODULE(_pico_snapshot_test, m) {
    m.def("inject", &injectPacket, pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("reset", []() {
        std::scoped_lock lock(leftMutex, rightMutex, leftHandMutex, rightHandMutex, bodyMutex, timestampMutex, handPacketDiagnosticMutex);
        LeftHandSnapshot = {}; RightHandSnapshot = {};
        LeftControllerSnapshot = {}; RightControllerSnapshot = {};
        LeftHandTrackingState = {}; RightHandTrackingState = {};
        LeftHandScale = RightHandScale = 1.0;
        LeftHandIsActive = RightHandIsActive = 0;
        BodyDataAvailable = false; BodyTimeStampNs = TimeStampNs = 0;
        HandPacketCount = 0; LastHandPacket = {}; HasLastHandPacket = false;
    });
    m.def("get_left_hand_snapshot", &getLeftHandSnapshot);
    m.def("get_right_hand_snapshot", &getRightHandSnapshot);
    m.def("get_hand_packet_diagnostics", &getHandPacketDiagnostics);
    m.def("get_left_controller_snapshot", &getLeftControllerSnapshot);
    m.def("get_right_controller_snapshot", &getRightControllerSnapshot);
    m.def("get_left_hand_tracking_state", &getLeftHandTrackingState);
    m.def("get_right_hand_tracking_state", &getRightHandTrackingState);
    m.def("get_left_hand_is_active", &getLeftHandIsActive);
    m.def("get_right_hand_is_active", &getRightHandIsActive);
    m.def("get_left_controller_pose", &getLeftControllerPose);
    m.def("get_right_controller_pose", &getRightControllerPose);
    m.def("is_body_data_available", &isBodyDataAvailable);
    m.def("get_body_timestamp_ns", &getBodyTimeStampNs);
}

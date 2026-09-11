#include <pybind11/pybind11.h>
#include <pybind11/chrono.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <thread>
#include <iostream>
#include <mutex>
#include <sstream>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include "PXREARobotSDK.h"


using json = nlohmann::json;

struct HandSnapshot {
    std::array<std::array<double, 7>, 26> pose{};
    std::array<uint64_t, 26> location_flags{};
    std::array<double, 26> radius{};
    double scale = 1.0;
    int32_t is_active = 0;
    int64_t source_timestamp_ns = 0;
    int32_t timestamp_source = 0;  // 0: device sample, 1: host monotonic content change.
    uint64_t binding_generation = 0;
};

HandSnapshot LeftHandSnapshot, RightHandSnapshot;

struct ControllerSnapshot {
    std::array<double, 7> pose{};
    bool primary_button = false, secondary_button = false, axis_click = false, menu_button = false;
    double grip = 0, trigger = 0;
    std::array<double, 2> axis{};
    uint64_t binding_generation = 0;
    int64_t receipt_timestamp_ns = 0;
};
ControllerSnapshot LeftControllerSnapshot, RightControllerSnapshot;

int64_t steadyTimeNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <size_t N>
bool strictPose(const json& value, std::array<double, N>& out) {
    if (!value.is_string()) return false;
    const auto text = value.get<std::string>();
    if (text.empty() || text.back() == ',') return false;
    std::stringstream ss(text);
    std::string item;
    size_t i = 0;
    while (std::getline(ss, item, ',')) {
        if (i == N || item.empty()) return false;
        size_t consumed = 0;
        const double number = std::stod(item, &consumed);
        if (item.find_first_not_of(" \t\r\n", consumed) != std::string::npos ||
            !std::isfinite(number)) return false;
        out[i++] = number;
    }
    return i == N;
}

bool parseControllerSnapshot(const json& c, ControllerSnapshot& out) {
    try {
        if (!c.is_object() || !strictPose(c.at("pose"), out.pose)) return false;
        out.primary_button = c.at("primaryButton").get<bool>();
        out.secondary_button = c.at("secondaryButton").get<bool>();
        out.menu_button = c.at("menuButton").get<bool>();
        out.axis_click = c.at("axisClick").get<bool>();
        out.axis = {c.at("axisX").get<double>(), c.at("axisY").get<double>()};
        out.trigger = c.at("trigger").get<double>();
        out.grip = c.at("grip").get<double>();
        return std::isfinite(out.axis[0]) && std::isfinite(out.axis[1]) &&
            std::isfinite(out.trigger) && std::isfinite(out.grip);
    } catch (const std::exception&) { return false; }
}

// Original APK packets omit the timestamp; complete flags/radii still provide
// optical evidence, with conservative per-side content-change freshness below.
// Validate any metadata that is present before changing even the legacy getters.
bool parseHandSnapshot(const json& hand, HandSnapshot& out, bool& optical) {
    try {
        if (!hand.is_object() || !hand.at("HandJointLocations").is_array() ||
            hand.at("HandJointLocations").size() != 26) return false;
        out.scale = hand.at("scale").get<double>();
        const auto& active = hand.at("isActive");
        if (!std::isfinite(out.scale) || !active.is_number_integer()) return false;
        if (active.is_number_unsigned()) {
            if (active.get<uint64_t>() > uint64_t(std::numeric_limits<int32_t>::max())) return false;
        } else if (active.get<int64_t>() < std::numeric_limits<int32_t>::min() ||
                   active.get<int64_t>() > std::numeric_limits<int32_t>::max()) return false;
        out.is_active = active.get<int32_t>();
        out.timestamp_source = hand.contains("timeStampNs") ? 0 : 1;
        if (out.timestamp_source == 0) {
            const auto& timestamp = hand.at("timeStampNs");
            if (!timestamp.is_number_integer() ||
                (timestamp.is_number_unsigned() &&
                 timestamp.get<uint64_t>() > uint64_t(std::numeric_limits<int64_t>::max()))) return false;
            out.source_timestamp_ns = timestamp.get<int64_t>();
            if (out.source_timestamp_ns <= 0) return false;
        }
        size_t flagCount = 0, radiusCount = 0;
        for (size_t i = 0; i < 26; ++i) {
            const auto& joint = hand.at("HandJointLocations")[i];
            if (!joint.is_object() || !strictPose(joint.at("p"), out.pose[i])) return false;
            if (joint.contains("s")) {
                const auto& flags = joint.at("s");
                if (flags.is_number_integer()) {
                    if (!flags.is_number_unsigned() && flags.get<int64_t>() < 0) return false;
                    out.location_flags[i] = flags.get<uint64_t>();
                } else if (flags.is_number_float()) {
                    const double value = flags.get<double>();
                    constexpr double maxSafeInteger = 9007199254740991.0;
                    if (!std::isfinite(value) || value < 0 || value > maxSafeInteger ||
                        std::trunc(value) != value) return false;
                    out.location_flags[i] = static_cast<uint64_t>(value);
                } else {
                    return false;
                }
                ++flagCount;
            }
            if (joint.contains("r")) {
                out.radius[i] = joint.at("r").get<double>();
                if (!std::isfinite(out.radius[i])) return false;
                ++radiusCount;
            }
        }
        if ((flagCount != 0 && flagCount != 26) ||
            (radiusCount != 0 && radiusCount != 26)) return false;
        optical = flagCount == 26 && radiusCount == 26;
        return true;
    } catch (const std::exception&) { return false; }
}

// Caller holds this side's mutex. Packet arrival alone never refreshes the
// original APK's cached side JSON. Once device timestamps are used, retain that
// requirement until binding reset rather than accepting a missing-field downgrade.
void commitHandSnapshot(HandSnapshot& current, HandSnapshot parsed) {
    if (parsed.timestamp_source == 1) {
        if (current.binding_generation != 0 && current.timestamp_source == 0) return;
        const bool unchanged = current.binding_generation != 0 &&
            parsed.pose == current.pose && parsed.location_flags == current.location_flags &&
            parsed.radius == current.radius && parsed.scale == current.scale &&
            parsed.is_active == current.is_active;
        parsed.source_timestamp_ns = unchanged ? current.source_timestamp_ns : steadyTimeNs();
    }
    parsed.binding_generation = current.binding_generation + 1;
    current = parsed;
}

std::array<double, 7> LeftControllerPose;
std::array<double, 7> RightControllerPose;
std::array<double, 7> HeadsetPose;

std::array<std::array<double, 7>, 26> LeftHandTrackingState;
double LeftHandScale = 1.0;
int LeftHandIsActive = 0;
std::array<std::array<double, 7>, 26> RightHandTrackingState;
double RightHandScale = 1.0;
int RightHandIsActive = 0;

// Whole body motion data - 24 joints for body tracking
std::array<std::array<double, 7>, 24> BodyJointsPose;  // Position and rotation for each joint
std::array<std::array<double, 6>, 24> BodyJointsVelocity;  // Velocity and angular velocity for each joint
std::array<std::array<double, 6>, 24> BodyJointsAcceleration;  // Acceleration and angular acceleration for each joint
std::array<int64_t, 24> BodyJointsTimestamp;  // IMU timestamp for each joint
int64_t BodyTimeStampNs = 0;  // Body data timestamp
bool BodyDataAvailable = false;  // Flag to indicate if body data is available

std::array<std::array<double, 7>, 3> MotionTrackerPose;  // Position and rotation for each joint
std::array<std::array<double, 6>, 3> MotionTrackerVelocity;  // Velocity and angular velocity for each joint
std::array<std::array<double, 6>, 3> MotionTrackerAcceleration;  // Acceleration and angular acceleration for each joint
std::array<std::string, 3> MotionTrackerSerialNumbers;  // Serial numbers of the motion trackers
int64_t MotionTimeStampNs = 0;  // Motion data timestamp
int NumMotionDataAvailable = 0;  // number of motion trackers


bool LeftMenuButton;
double LeftTrigger;
double LeftGrip;
std::array<double, 2> LeftAxis{0.0, 0.0};
bool LeftAxisClick;
bool LeftPrimaryButton;
bool LeftSecondaryButton;

bool RightMenuButton;
double RightTrigger;
double RightGrip;
std::array<double, 2> RightAxis{0.0, 0.0};
bool RightAxisClick;
bool RightPrimaryButton;
bool RightSecondaryButton;

int64_t TimeStampNs;

std::mutex leftMutex;
std::mutex rightMutex;
std::mutex headsetPoseMutex;
std::mutex timestampMutex;
std::mutex leftHandMutex;
std::mutex rightHandMutex;
std::mutex bodyMutex;  // Mutex for body tracking data
std::mutex motionMutex;
std::mutex handPacketDiagnosticMutex;
uint64_t HandPacketCount = 0;
json LastHandPacket;
bool HasLastHandPacket = false;



std::array<double, 7> stringToPoseArray(const std::string& poseStr) {
    std::array<double, 7> result{0};
    std::stringstream ss(poseStr);
    std::string value;
    int i = 0;
    while (std::getline(ss, value, ',') && i < 7) {
        result[i++] = std::stod(value);
    }
    return result;
}

std::array<double, 6> stringToVelocityArray(const std::string& velocityStr) {
    std::array<double, 6> result{0};
    std::stringstream ss(velocityStr);
    std::string value;
    int i = 0;
    while (std::getline(ss, value, ',') && i < 6) {
        result[i++] = std::stod(value);
    }
    return result;
}

void OnPXREAClientCallback(void* context, PXREAClientCallbackType type, int status, void* userData)
{
    switch (type)
    {
    case PXREAServerConnect:
        std::cout << "server connect\n" << std::endl;
        break;
    case PXREAServerDisconnect:
        std::cout << "server disconnect\n" << std::endl;
        break;
    case PXREADeviceFind:
        std::cout << "device found\n" << (const char*)userData << std::endl;
        break;
    case PXREADeviceMissing:
        std::cout << "device missing\n" << (const char*)userData << std::endl;
        break;
    case PXREADeviceConnect:
        std::cout << "device connect\n" << (const char*)userData << status << std::endl;
        break;
    case PXREADeviceStateJson:
        auto& dsj = *((PXREADevStateJson*)userData);
        

        try {
            json data = json::parse(dsj.stateJson);
            if (data.contains("value")) {
                auto value = json::parse(data["value"].get<std::string>());
                {
                    std::lock_guard<std::mutex> lock(handPacketDiagnosticMutex);
                    ++HandPacketCount;
                    if (value.contains("Hand")) {
                        LastHandPacket = value["Hand"];
                        HasLastHandPacket = true;
                    } else {
                        LastHandPacket = {};
                        HasLastHandPacket = false;
                    }
                }
                if (value.contains("Controller") && value["Controller"].is_object() &&
                    value["Controller"].contains("left")) {
                    ControllerSnapshot parsed;
                    if (parseControllerSnapshot(value["Controller"]["left"], parsed)) {
                        std::lock_guard<std::mutex> lock(leftMutex);
                        parsed.binding_generation = LeftControllerSnapshot.binding_generation + 1;
                        parsed.receipt_timestamp_ns = steadyTimeNs();
                        LeftControllerSnapshot = parsed;
                        LeftControllerPose = parsed.pose;
                        LeftTrigger = parsed.trigger;
                        LeftGrip = parsed.grip;
                        LeftMenuButton = parsed.menu_button;
                        LeftAxis = parsed.axis;
                        LeftAxisClick = parsed.axis_click;
                        LeftPrimaryButton = parsed.primary_button;
                        LeftSecondaryButton = parsed.secondary_button;
                    }
                }
                if (value.contains("Controller") && value["Controller"].is_object() &&
                    value["Controller"].contains("right")) {
                    ControllerSnapshot parsed;
                    if (parseControllerSnapshot(value["Controller"]["right"], parsed)) {
                        std::lock_guard<std::mutex> lock(rightMutex);
                        parsed.binding_generation = RightControllerSnapshot.binding_generation + 1;
                        parsed.receipt_timestamp_ns = steadyTimeNs();
                        RightControllerSnapshot = parsed;
                        RightControllerPose = parsed.pose;
                        RightTrigger = parsed.trigger;
                        RightGrip = parsed.grip;
                        RightMenuButton = parsed.menu_button;
                        RightAxis = parsed.axis;
                        RightAxisClick = parsed.axis_click;
                        RightPrimaryButton = parsed.primary_button;
                        RightSecondaryButton = parsed.secondary_button;
                    }
                }
                if (value.contains("Hand") && value["Hand"].is_object() &&
                    value["Hand"].contains("leftHand")) {
                    HandSnapshot parsed;
                    bool optical = false;
                    if (parseHandSnapshot(value["Hand"]["leftHand"], parsed, optical)) {
                        std::lock_guard<std::mutex> lock(leftHandMutex);
                        if (optical) {
                            commitHandSnapshot(LeftHandSnapshot, parsed);
                        }
                        LeftHandScale = parsed.scale;
                        LeftHandIsActive = parsed.is_active;
                        LeftHandTrackingState = parsed.pose;
                    }
                }
                if (value.contains("Hand") && value["Hand"].is_object() &&
                    value["Hand"].contains("rightHand")) {
                    HandSnapshot parsed;
                    bool optical = false;
                    if (parseHandSnapshot(value["Hand"]["rightHand"], parsed, optical)) {
                        std::lock_guard<std::mutex> lock(rightHandMutex);
                        if (optical) {
                            commitHandSnapshot(RightHandSnapshot, parsed);
                        }
                        RightHandScale = parsed.scale;
                        RightHandIsActive = parsed.is_active;
                        RightHandTrackingState = parsed.pose;
                    }
                }
                // A malformed auxiliary field must not suppress independently valid sides/body.
                try {
                    if (value.contains("Head")) {
                        std::array<double, 7> pose;
                        if (strictPose(value["Head"].at("pose"), pose)) {
                            std::lock_guard<std::mutex> lock(headsetPoseMutex);
                            HeadsetPose = pose;
                        }
                    }
                } catch (const std::exception&) {}
                try {
                    if (value.contains("timeStampNs")) {
                        const auto timestamp = value["timeStampNs"].get<int64_t>();
                        std::lock_guard<std::mutex> lock(timestampMutex);
                        TimeStampNs = timestamp;
                    }
                } catch (const std::exception&) {}
                // Parse Body data for whole body motion capture
                if (value.contains("Body")) {
                    auto& body = value["Body"];
                    {
                        std::lock_guard<std::mutex> lock(bodyMutex);
                        
                        if (body.contains("timeStampNs")) {
                            BodyTimeStampNs = body["timeStampNs"].get<int64_t>();
                        }
                        
                        if (body.contains("joints") && body["joints"].is_array()) {
                            auto joints = body["joints"];
                            int jointCount = std::min(static_cast<int>(joints.size()), 24);
                            
                            for (int i = 0; i < jointCount; i++) {
                                auto& joint = joints[i];
                                
                                // Parse pose (position and rotation)
                                if (joint.contains("p")) {
                                    BodyJointsPose[i] = stringToPoseArray(joint["p"].get<std::string>());
                                }
                                
                                // Parse velocity and angular velocity
                                if (joint.contains("va")) {
                                    BodyJointsVelocity[i] = stringToVelocityArray(joint["va"].get<std::string>());
                                }
                                
                                // Parse acceleration and angular acceleration
                                if (joint.contains("wva")) {
                                    BodyJointsAcceleration[i] = stringToVelocityArray(joint["wva"].get<std::string>());
                                }
                                
                                // Parse IMU timestamp
                                if (joint.contains("t")) {
                                    BodyJointsTimestamp[i] = joint["t"].get<int64_t>();
                                }
                            }
                            
                            BodyDataAvailable = true;
                        }
                    }
                }
                //parse individual tracker data
                if (value.contains("Motion")) {
                    auto& motion = value["Motion"];
                    {
                        std::lock_guard<std::mutex> lock(motionMutex);
                        if (motion.contains("timeStampNs")) {
                            MotionTimeStampNs = motion["timeStampNs"].get<int64_t>();
                        }
                        if (motion.contains("joints") && motion["joints"].is_array()) {
                            auto joints = motion["joints"];
                            NumMotionDataAvailable = std::min(static_cast<int>(joints.size()), 3);

                            for (int i = 0; i < NumMotionDataAvailable; i++) {
                                auto& joint = joints[i];

                                // Parse pose (position and rotation)
                                if (joint.contains("p")) {
                                    MotionTrackerPose[i] = stringToPoseArray(joint["p"].get<std::string>());
                                }

                                // Parse velocity and angular velocity
                                if (joint.contains("va")) {
                                    MotionTrackerVelocity[i] = stringToVelocityArray(joint["va"].get<std::string>());
                                }

                                // Parse acceleration and angular acceleration
                                if (joint.contains("wva")) {
                                    MotionTrackerAcceleration[i] = stringToVelocityArray(joint["wva"].get<std::string>());
                                }

                                if (joint.contains("sn")) {
                                    MotionTrackerSerialNumbers[i] = joint["sn"].get<std::string>();
                                }
                            }

                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "JSON parsing error: " << e.what() << std::endl;
        }
            break;
    }
}

void init() {
    if (PXREAInit(NULL, OnPXREAClientCallback, PXREAFullMask) != 0) {
        throw std::runtime_error("PXREAInit failed");
    }
}

void deinit() {
    PXREADeinit();
}

std::array<double, 7> getLeftControllerPose() {
    std::lock_guard<std::mutex> lock(leftMutex);
    return LeftControllerPose;
}

std::array<double, 7> getRightControllerPose() {
    std::lock_guard<std::mutex> lock(rightMutex);
    return RightControllerPose;
}

std::array<double, 7> getHeadsetPose() {
    std::lock_guard<std::mutex> lock(headsetPoseMutex);
    return HeadsetPose;
}

double getLeftTrigger() {
    std::lock_guard<std::mutex> lock(leftMutex);
    return LeftTrigger;
}

double getLeftGrip() {
    std::lock_guard<std::mutex> lock(leftMutex);
    return LeftGrip;
}

double getRightTrigger() {
    std::lock_guard<std::mutex> lock(rightMutex);
    return RightTrigger;
}

double getRightGrip() {
    std::lock_guard<std::mutex> lock(rightMutex);
    return RightGrip;
}

bool getLeftMenuButton() {
    std::lock_guard<std::mutex> lock(leftMutex);
    return LeftMenuButton;
}

bool getRightMenuButton() {
    std::lock_guard<std::mutex> lock(rightMutex);
    return RightMenuButton;
}

bool getLeftAxisClick() {
    std::lock_guard<std::mutex> lock(leftMutex);
    return LeftAxisClick;
}

bool getRightAxisClick() {
    std::lock_guard<std::mutex> lock(rightMutex);
    return RightAxisClick;
}

std::array<double, 2> getLeftAxis() {
    std::lock_guard<std::mutex> lock(leftMutex);
    return LeftAxis;
}


std::array<double, 2> getRightAxis() {
    std::lock_guard<std::mutex> lock(rightMutex);
    return RightAxis;
}

bool getLeftPrimaryButton() {
    std::lock_guard<std::mutex> lock(leftMutex);
    return LeftPrimaryButton;
}

bool getRightPrimaryButton() {
    std::lock_guard<std::mutex> lock(rightMutex);
    return RightPrimaryButton;
}

bool getLeftSecondaryButton() {
    std::lock_guard<std::mutex> lock(leftMutex);
    return LeftSecondaryButton;
}

bool getRightSecondaryButton() {
    std::lock_guard<std::mutex> lock(rightMutex);
    return RightSecondaryButton;
}

int64_t getTimeStampNs() {
    std::lock_guard<std::mutex> lock(timestampMutex);
    return TimeStampNs;
}

std::array<std::array<double, 7>, 26> getLeftHandTrackingState() {
    std::lock_guard<std::mutex> lock(leftHandMutex);
    return LeftHandTrackingState;
}

int getLeftHandScale() {
    std::lock_guard<std::mutex> lock(leftHandMutex);
    return LeftHandScale;
}

int getLeftHandIsActive() {
    std::lock_guard<std::mutex> lock(leftHandMutex);
    return LeftHandIsActive;
}

std::array<std::array<double, 7>, 26> getRightHandTrackingState() {
    std::lock_guard<std::mutex> lock(rightHandMutex);
    return RightHandTrackingState;
}

int getRightHandScale() {
    std::lock_guard<std::mutex> lock(rightHandMutex);
    return RightHandScale;
}

int getRightHandIsActive() {
    std::lock_guard<std::mutex> lock(rightHandMutex);
    return RightHandIsActive;
}

pybind11::dict handSnapshot(const HandSnapshot& snapshot) {
    namespace py = pybind11;
    py::array_t<double> pose({26, 7});
    auto poseView = pose.mutable_unchecked<2>();
    for (ssize_t i = 0; i < 26; ++i) for (ssize_t j = 0; j < 7; ++j) poseView(i, j) = snapshot.pose[i][j];
    py::array_t<uint64_t> flags(26);
    py::array_t<double> radius(26);
    auto flagsView = flags.mutable_unchecked<1>();
    auto radiusView = radius.mutable_unchecked<1>();
    for (ssize_t i = 0; i < 26; ++i) { flagsView(i) = snapshot.location_flags[i]; radiusView(i) = snapshot.radius[i]; }
    py::dict result;
    result["pose"] = pose;
    result["location_flags"] = flags;
    result["radius"] = radius;
    result["scale"] = snapshot.scale;
    result["is_active"] = snapshot.is_active;
    result["source_timestamp_ns"] = snapshot.source_timestamp_ns;
    result["timestamp_source"] = snapshot.timestamp_source;
    result["binding_generation"] = snapshot.binding_generation;
    return result;
}

pybind11::dict getLeftHandSnapshot() {
    HandSnapshot snapshot;
    { std::lock_guard<std::mutex> lock(leftHandMutex); snapshot = LeftHandSnapshot; }
    return handSnapshot(snapshot);
}

pybind11::dict getRightHandSnapshot() {
    HandSnapshot snapshot;
    { std::lock_guard<std::mutex> lock(rightHandMutex); snapshot = RightHandSnapshot; }
    return handSnapshot(snapshot);
}

pybind11::dict getHandPacketDiagnostics() {
    std::lock_guard<std::mutex> lock(handPacketDiagnosticMutex);
    pybind11::dict result;
    result["packets"] = HandPacketCount;
    result["hand_json"] = HasLastHandPacket ? LastHandPacket.dump() : "";
    return result;
}

pybind11::dict controllerSnapshot(const ControllerSnapshot& s) {
    namespace py = pybind11;
    py::dict d;
    d["pose"] = py::cast(s.pose);
    d["primary_button"] = s.primary_button; d["secondary_button"] = s.secondary_button;
    d["grip"] = s.grip; d["trigger"] = s.trigger; d["axis"] = py::make_tuple(s.axis[0], s.axis[1]);
    d["axis_click"] = s.axis_click; d["menu_button"] = s.menu_button;
    d["binding_generation"] = s.binding_generation;
    d["receipt_age_ns"] = s.binding_generation == 0 ? std::numeric_limits<int64_t>::max() :
        std::max<int64_t>(0, steadyTimeNs() - s.receipt_timestamp_ns);
    return d;
}

pybind11::dict getLeftControllerSnapshot() {
    ControllerSnapshot snapshot;
    { std::lock_guard<std::mutex> lock(leftMutex); snapshot = LeftControllerSnapshot; }
    return controllerSnapshot(snapshot);
}

pybind11::dict getRightControllerSnapshot() {
    ControllerSnapshot snapshot;
    { std::lock_guard<std::mutex> lock(rightMutex); snapshot = RightControllerSnapshot; }
    return controllerSnapshot(snapshot);
}

// Body tracking functions
bool isBodyDataAvailable() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    return BodyDataAvailable;
}

std::array<std::array<double, 7>, 24> getBodyJointsPose() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    return BodyJointsPose;
}

std::array<std::array<double, 6>, 24> getBodyJointsVelocity() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    return BodyJointsVelocity;
}

std::array<std::array<double, 6>, 24> getBodyJointsAcceleration() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    return BodyJointsAcceleration;
}

std::array<int64_t, 24> getBodyJointsTimestamp() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    return BodyJointsTimestamp;
}

int64_t getBodyTimeStampNs() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    return BodyTimeStampNs;
}

int numMotionDataAvailable() {
    std::lock_guard<std::mutex> lock(motionMutex);
    return NumMotionDataAvailable;
}

std::vector<std::array<double, 7>> getMotionTrackerPose() {
    std::lock_guard<std::mutex> lock(motionMutex);
    std::vector<std::array<double, 7>> result;
    for (int i = 0; i < NumMotionDataAvailable; i++) {
        result.push_back(MotionTrackerPose[i]);
    }
    return result;
}

std::vector<std::array<double, 6>> getMotionTrackerVelocity() {
    std::lock_guard<std::mutex> lock(motionMutex);
    std::vector<std::array<double, 6>> result;
    for (int i = 0; i < NumMotionDataAvailable; i++) {
        result.push_back(MotionTrackerVelocity[i]);
    }
    return result;
}

std::vector<std::array<double, 6>> getMotionTrackerAcceleration() {
    std::lock_guard<std::mutex> lock(motionMutex);
    std::vector<std::array<double, 6>> result;
    for (int i = 0; i < NumMotionDataAvailable; i++) {
        result.push_back(MotionTrackerAcceleration[i]);
    }
    return result;
}

std::vector<std::string> getMotionTrackerSerialNumbers() {
    std::lock_guard<std::mutex> lock(motionMutex);
    std::vector<std::string> result;
    for (int i = 0; i < NumMotionDataAvailable; i++) {
        result.push_back(MotionTrackerSerialNumbers[i]);
    }
    return result;
}

int64_t getMotionTimeStampNs() {
    std::lock_guard<std::mutex> lock(motionMutex);
    return MotionTimeStampNs;
}

int DeviceControlJsonWrapper(const std::string& dev_id, const std::string& json_str) {
    const int rc = PXREADeviceControlJson(dev_id.c_str(), json_str.c_str());
    if (rc != 0) {
        throw std::runtime_error("device_control_json failed");
    }
    return rc; // 0
}

int SendBytesToDeviceWrapper(const std::string& dev_id, pybind11::bytes blob) {
    std::string s = blob;  // copy Python bytes to std::string
    const int rc = PXREASendBytesToDevice(dev_id.c_str(), s.data(), static_cast<unsigned>(s.size()));
    if (rc != 0) {
        throw std::runtime_error("send_bytes_to_device failed");
    }
    return rc; // 0
}


PYBIND11_MODULE(xrobotoolkit_sdk, m) {
    m.def("init", &init, "Initialize the PXREARobot SDK.");
    m.def("close", &deinit, "Deinitialize the PXREARobot SDK.");
    m.def("get_left_controller_pose", &getLeftControllerPose, "Get the left controller pose.");
    m.def("get_right_controller_pose", &getRightControllerPose, "Get the right controller pose.");
    m.def("get_headset_pose", &getHeadsetPose, "Get the headset pose.");
    m.def("get_left_trigger", &getLeftTrigger, "Get the left trigger value.");
    m.def("get_left_grip", &getLeftGrip, "Get the left grip value.");
    m.def("get_right_trigger", &getRightTrigger, "Get the right trigger value.");
    m.def("get_right_grip", &getRightGrip, "Get the right grip value.");
    m.def("get_left_menu_button", &getLeftMenuButton, "Get the left menu button state.");
    m.def("get_right_menu_button", &getRightMenuButton, "Get the right menu button state.");
    m.def("get_left_axis_click", &getLeftAxisClick, "Get the left axis click state.");
    m.def("get_right_axis_click", &getRightAxisClick, "Get the right axis click state.");
    m.def("get_left_axis", &getLeftAxis, "Get the left axis values (x, y).");
    m.def("get_right_axis", &getRightAxis, "Get the right axis values (x, y).");
    m.def("get_X_button", &getLeftPrimaryButton, "Get the left primary button state.");
    m.def("get_A_button", &getRightPrimaryButton, "Get the right primary button state.");
    m.def("get_Y_button", &getLeftSecondaryButton, "Get the left secondary button state.");
    m.def("get_B_button", &getRightSecondaryButton, "Get the right secondary button state.");
    m.def("get_time_stamp_ns", &getTimeStampNs, "Get the timestamp in nanoseconds.");
    m.def("get_left_hand_tracking_state", &getLeftHandTrackingState, "Get the left hand state.");
    m.def("get_right_hand_tracking_state", &getRightHandTrackingState, "Get the right hand state.");
    m.def("get_left_hand_is_active", &getLeftHandIsActive, "Get the left hand tracking quality (0 = low, 1 = high).");
    m.def("get_right_hand_is_active", &getRightHandIsActive, "Get the right hand tracking quality (0 = low, 1 = high).");
    m.def("get_left_hand_snapshot", &getLeftHandSnapshot, "Get one atomic left hand snapshot.");
    m.def("get_right_hand_snapshot", &getRightHandSnapshot, "Get one atomic right hand snapshot.");
    m.def("get_hand_packet_diagnostics", &getHandPacketDiagnostics, "Get native hand packet diagnostics.");
    m.def("get_left_controller_snapshot", &getLeftControllerSnapshot, "Get one atomic left controller snapshot.");
    m.def("get_right_controller_snapshot", &getRightControllerSnapshot, "Get one atomic right controller snapshot.");
    
    // Body tracking functions
    m.def("is_body_data_available", &isBodyDataAvailable, "Check if body tracking data is available.");
    m.def("get_body_joints_pose", &getBodyJointsPose, "Get the body joints pose data (24 joints, 7 values each: x,y,z,qx,qy,qz,qw).");
    m.def("get_body_joints_velocity", &getBodyJointsVelocity, "Get the body joints velocity data (24 joints, 6 values each: vx,vy,vz,wx,wy,wz).");
    m.def("get_body_joints_acceleration", &getBodyJointsAcceleration, "Get the body joints acceleration data (24 joints, 6 values each: ax,ay,az,wax,way,waz).");
    m.def("get_body_joints_timestamp", &getBodyJointsTimestamp, "Get the body joints IMU timestamp data (24 joints).");
    m.def("get_body_timestamp_ns", &getBodyTimeStampNs, "Get the body data timestamp in nanoseconds.");

    // Motion tracker functions
    m.def("num_motion_data_available", &numMotionDataAvailable, "Check if motion tracker data is available.");
    m.def("get_motion_tracker_pose", &getMotionTrackerPose, "Get the motion tracker pose data (3 trackers, 7 values each: x,y,z,qx,qy,qz,qw).");
    m.def("get_motion_tracker_velocity", &getMotionTrackerVelocity, "Get the motion tracker velocity data (3 trackers, 6 values each: vx,vy,vz,wx,wy,wz).");
    m.def("get_motion_tracker_acceleration", &getMotionTrackerAcceleration, "Get the motion tracker acceleration data (3 trackers, 6 values each: ax,ay,az,wax,way,waz).");
    m.def("get_motion_tracker_serial_numbers", &getMotionTrackerSerialNumbers, "Get the serial numbers of the motion trackers.");
    m.def("get_motion_timestamp_ns", &getMotionTimeStampNs, "Get the motion data timestamp in nanoseconds.");
    

    // send json bytes functions
    m.def("device_control_json", &DeviceControlJsonWrapper, "Send a JSON control command to a device");
    m.def("send_bytes_to_device", &SendBytesToDeviceWrapper, "Send raw bytes to a device");
    
    m.doc() = "Python bindings for PXREARobot SDK using pybind11.";
}

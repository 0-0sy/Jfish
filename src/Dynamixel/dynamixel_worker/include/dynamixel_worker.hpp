#ifndef DYNAMIXEL_WORKER_HPP
#define DYNAMIXEL_WORKER_HPP

#include <cstdio>
#include <memory>
#include <string>
#include <array>
#include "rclcpp/rclcpp.hpp"
#include "watchdog_interfaces/msg/node_state.hpp"
#include "mujoco_interfaces/msg/mu_jo_co_meas.hpp"
#include "dynamixel_interfaces/msg/joint_val.hpp"
#include "dynamixel_sdk/dynamixel_sdk.h"

#include <iostream>
#include <chrono>
#include <stdexcept>

// Dynamixel Addresses
#define ADDR_OPERATING_MODE      11
#define ADDR_TORQUE_ENABLE       64
#define ADDR_GOAL_POSITION       116
#define ADDR_PRESENT_POSITION    132
#define ADDR_POSITION_P_GAIN     84
#define ADDR_POSITION_I_GAIN     82
#define ADDR_POSITION_D_GAIN     80
#define ADDR_VELOCITY_P_GAIN      78
#define ADDR_VELOCITY_I_GAIN      76

#define PROTOCOL_VERSION 2.0
#define BAUDRATE         4000000
#define DEVICE_NAME      "/dev/ttyUSB1"

#define ARM_NUM 1

constexpr std::array<std::array<uint8_t, 5>, 4> DXL_IDS = {{
  {1, 2, 3, 4, 5},   // Arm 1
  {6, 7, 8, 9, 10},  // Arm 2
  {11, 12, 13, 14, 15}, // Arm 3
  {16, 17, 18, 19, 20}  // Arm 4
}};

constexpr double PI = 3.1415926535897932384626433832706;
constexpr double rad2ppr_J1 = 6.25 * 2048.0 / PI;
constexpr double ppr2rad_J1 = PI / 2048.0 / 6.25;
constexpr double rad2ppr = 2048.0 / PI;
constexpr double ppr2rad = PI / 2048.0;

class DynamixelNode : public rclcpp::Node {
public:
  DynamixelNode(const std::string &device_name);
  virtual ~DynamixelNode();
  bool init_Dynamixel();

private:
  void Dynamixel_Write_Read();
  void Mujoco_Pub();

  void armchanger_callback(const dynamixel_interfaces::msg::JointVal::SharedPtr msg);
  void mujoco_callback(const mujoco_interfaces::msg::MuJoCoMeas::SharedPtr msg);
  void change_position_gain(uint8_t dxl_id, uint16_t p_gain, uint16_t i_gain, uint16_t d_gain);
  void change_velocity_gain(uint8_t dxl_id, uint16_t p_gain, uint16_t i_gain);
  void heartbeat_timer_callback();

  // ROS2 communication
  rclcpp::Subscription<dynamixel_interfaces::msg::JointVal>::SharedPtr joint_val_subscriber_;
  rclcpp::Subscription<mujoco_interfaces::msg::MuJoCoMeas>::SharedPtr mujoco_subscriber_;
  rclcpp::Publisher<watchdog_interfaces::msg::NodeState>::SharedPtr heartbeat_publisher_;
  rclcpp::Publisher<dynamixel_interfaces::msg::JointVal>::SharedPtr mujoco_publisher_;
  rclcpp::Publisher<dynamixel_interfaces::msg::JointVal>::SharedPtr pos_mea_publisher_;
  rclcpp::TimerBase::SharedPtr motor_timer_;
  rclcpp::TimerBase::SharedPtr heartbeat_timer_;

  // Dynamixel SDK objects as class member variables
  dynamixel::PortHandler* portHandler_;
  dynamixel::PacketHandler* packetHandler_;
  dynamixel::GroupSyncWrite* groupSyncWrite_;
  dynamixel::GroupSyncRead*  groupSyncRead_;

  // allocator sub
  int arm_des[4][5] = { // [ppr]
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0}
  };

  // mujoco or dynamixel read
  double arm_mea[4][5] = { // [rad]
    {0., -0.84522, 1.50944, 0.90812, 0.},
    {0., -0.84522, 1.50944, 0.90812, 0.},
    {0., -0.84522, 1.50944, 0.90812, 0.},
    {0., -0.84522, 1.50944, 0.90812, 0.}
  };

  uint16_t dnmxl_err_cnt_ = 0;
  uint8_t heartbeat_state_;
};

#endif // DYNAMIXEL_WORKER_HPP
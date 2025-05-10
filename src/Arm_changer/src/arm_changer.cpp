#include "arm_changer.hpp"

using namespace std::chrono_literals;

inline double map_value(double input, double in_min, double in_max, double out_min, double out_max) {
  return (input - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

ArmChangerWorker::ArmChangerWorker(): Node("arm_changing_node") {
  // ROS2 Subscribers
  sbus_subscription_ = this->create_subscription<sbus_interfaces::msg::SbusSignal>("/sbus_signal", 1, std::bind(&ArmChangerWorker::sbus_callback, this, std::placeholders::_1));
  killcmd_subscription_ = this->create_subscription<sbus_interfaces::msg::KillCmd>("sbus_kill", 1, std::bind(&ArmChangerWorker::killCmd_callback, this, std::placeholders::_1));

  // ROS2 Publisher
  joint_publisher_ = this->create_publisher<dynamixel_interfaces::msg::JointVal>("/joint_cmd", 1);
  heartbeat_publisher_ = this->create_publisher<watchdog_interfaces::msg::NodeState>("/armchanger_state", 1);

  // ROS2 Timer
  joint_timer_ = this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&ArmChangerWorker::joint_callback, this));
  heartbeat_timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&ArmChangerWorker::heartbeat_timer_callback, this));

  a1_q.resize(5);
  a2_q.resize(5);
  a3_q.resize(5);
  a4_q.resize(5);

  // workspace constrain
  x_min_ = -270.; 
  x_max_ = -330.;
  yaw_min_ = -0.7854;
  yaw_max_ = 0.7854;
  y_fixed_ = 0.; 
  z_min_ = 50.;
  z_max_ = 220.;

  // initial handshake: immediately send 42 and enable subsequent heartbeat
  hb_state_   = 42;
  hb_enabled_ = true;
}

void ArmChangerWorker::sbus_callback(const sbus_interfaces::msg::SbusSignal::SharedPtr msg) {
  double x = map_value(static_cast<double>(msg->ch[10]), 352, 1696, x_min_, x_max_);
  double z = map_value(static_cast<double>(msg->ch[11]), 352, 1696, z_min_, z_max_);
  th1_ = map_value(static_cast<double>(msg->ch[1]), 352, 1696, yaw_min_, yaw_max_);

  // RCLCPP_INFO(this->get_logger(), "x: %.2f, z: %.2f", x, z);
  compute_ik(x, y_fixed_, z, heading_fixed_);
}

void ArmChangerWorker::killCmd_callback(const sbus_interfaces::msg::KillCmd::SharedPtr msg) {
  kill_activated_ = msg->kill_activated;
  // RCLCPP_INFO(this->get_logger(), "kill_activated_: %s", kill_activated_ ? "true" : "false");
}


void ArmChangerWorker::compute_ik(const double x, const double y, const double z, const Eigen::Vector3d &heading){
  Eigen::Vector3d position(x, y, z);
  Eigen::Vector3d p05 = position;
  Eigen::Vector3d p04 = p05 - a5_ * heading;

  double th1_ = -std::atan2(p04(0), p04(1)) - PI / 2;

  double n = p04(1) * heading(0) - p04(0) * heading(1);
  th5_ = std::acos(std::abs(n) / std::sqrt(std::pow(p04(1), 2) + std::pow(p04(0), 2)));
  if (th5_ <= PI / 2) th5_ -= PI / 2;
  if (p04(0) * p05(1) - p04(1) * p05(0) > 0) th5_ = -th5_;

  double cos_1 = std::cos(th1_);
  double sin_1 = std::sin(th1_);
  Eigen::Vector3d heading_projected = heading - std::sin(th5_) * Eigen::Vector3d(sin_1, -cos_1, 0);
  Eigen::Vector3d p34 = a4_ * heading_projected / heading_projected.norm();
  Eigen::Vector3d p03 = p04 - p34;

  Eigen::Vector3d p01(-a1_ * cos_1, -a1_ * sin_1, 0);
  double x_prime = std::sqrt(std::pow(p01(0) - p03(0), 2) + std::pow(p01(1), 2));
  double y_prime = p03(2);
  double xy_sqr_sum = std::pow(x_prime, 2) + std::pow(y_prime, 2);

  double cos_3 = (xy_sqr_sum - (a2_ * a2_ + a3_ * a3_)) / (2 * a2_ * a3_);
  double sin_3 = std::sqrt(1 - std::pow(cos_3, 2));
  th3_ = -std::acos(cos_3);

  if (p03(0) < p01(0)) {
      th2_ = -std::atan2(y_prime, x_prime) + std::atan2(a3_ * sin_3, a2_ + a3_ * cos_3);
  } else {
      th2_ = std::atan2(y_prime, x_prime) + std::atan2(a3_ * sin_3, a2_ + a3_ * cos_3) - PI;
  }

  double cos_2 = std::cos(th2_);
  Eigen::Vector3d p02 = p01 - a2_ * Eigen::Vector3d(cos_2 * cos_1, cos_2 * sin_1, std::sin(th2_));
  Eigen::Vector3d p32 = p02 - p03;

  double cos_4 = std::clamp(p32.dot(p34) / (a3_ * a4_), -1.0, 1.0);
  th4_ = PI - std::acos(cos_4);
  if (std::abs(cos_4) == 1) th4_ = 0.0;

  double th4_ref = std::atan2(p34(2), p34(0)) - std::atan2(p32(2), p32(0));
  if (th4_ref < 0) th4_ref += 2 * PI;
  if (th4_ref > PI) th4_ = -th4_;
}

void ArmChangerWorker::joint_callback() {
  auto joint_msg = dynamixel_interfaces::msg::JointVal();

  joint_msg.a1_des = {th1_, th2_, th3_, th4_, th5_};
  joint_msg.a2_des = {th1_, th2_, th3_, th4_, th5_};
  joint_msg.a3_des = {th1_, th2_, th3_, th4_, th5_};
  joint_msg.a4_des = {th1_, th2_, th3_, th4_, th5_};

  RCLCPP_INFO(this->get_logger(), "%.2f, %.2f, %.2f, %.2f, %.2f", th1_, th2_, th3_, th4_, th5_);

  joint_publisher_->publish(joint_msg);
}

void ArmChangerWorker::watchdog_callback(const watchdog_interfaces::msg::NodeState::SharedPtr msg) {
  // Watchdog update
  watchdog_state_ = msg->state;
}

void ArmChangerWorker::heartbeat_timer_callback() {
  // gate until handshake done
  if (!hb_enabled_) {return;}

  watchdog_interfaces::msg::NodeState state_msg;
  state_msg.state = hb_state_;
  heartbeat_publisher_->publish(state_msg);

  // uint8 overflow wraps automatically
  hb_state_ = static_cast<uint8_t>(hb_state_ + 1);
}
  
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArmChangerWorker>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

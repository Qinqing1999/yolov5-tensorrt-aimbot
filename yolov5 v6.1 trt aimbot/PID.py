class PID:
    def __init__(self, dt, max, min, Kp, Ki, Kd):
        self.dt = dt  # 循环时长
        self.max = max  # 操作变量最大值
        self.min = min  # 操作变量最小值
        self.Kp = Kp  # 比例增益
        self.Ki = Ki  # 积分增益
        self.Kd = Kd  # 微分增益
        self.integral = 0  # 直到上一次的误差值
        self.pre_error = 0  # 上一次的误差值

    def calculate(self, setPoint, pv):
        # 其中 pv:process value 即过程值，
        error = setPoint - pv  # 误差
        Pout = self.Kp * error  # 比例项
        self.integral += error * self.dt
        Iout = self.Ki * self.integral  # 积分项
        derivative = (error - self.pre_error) / self.dt
        Dout = self.Kd * derivative  # 微分项

        output = Pout + Iout + Dout  # 新的目标值

        if (output > self.max):
            output = self.max
        elif (output < self.min):
            output = self.min

        self.pre_error = error  # 保存本次误差，以供下次计算
        return output



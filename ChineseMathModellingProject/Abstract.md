# Abstract

Wave energy is one of the most important energy sources in the ocean, characterized by its pollution-free nature and high storage capacity. It effectively mitigates energy crises and reduces pollution. This study establishes a numerical model for wave energy devices, derives the motion equations, and designs for maximum power efficiency.

## Problem 1
First, based on the assumption that the float maintains a circular trajectory, the float’s buoyancy is considered equivalent to a circular float with 1/3 the height. The force analysis and motion equations are derived by relating its displacement, velocity, and acceleration. By applying Newton’s second law, the motion equation is corrected for wave damping effects. Subsequently, a numerical method is used to identify the most suitable corrected model for wave motion resistance, yielding the final optimal model.

## Problem 2
Using the revised model from Problem 1, the instantaneous power and average power are calculated for cases where the damping force is constant. A step-size reduction method is used to find the optimal power level and precision, resulting in a maximum power of **28.3068W** and a corresponding optimal damping force of **18310 N·s/m**. For cases with variable damping force, the coupling between damping and velocity is analyzed. A steady-state condition yields a maximum power of **275.3W**, with a ratio of **100000**.

## Problem 3
The movement of the float in horizontal and vertical directions is analyzed under the equivalent floating model. The hydrodynamic equations are decomposed into horizontal and vertical components, with forces analyzed separately. Using trigonometric laws and angular velocity concepts, the equations of motion are established.

## Problem 4
A functional conversion relationship is analyzed, where the output power equals the combined work of the direct damping force and the rotational damping torque. Instantaneous power and average power are calculated for cases where damping is time-varying. After selecting an appropriate step size, maximum power and optimal damping are determined through optimization, achieving a maximum power output of **7.053W**, a direct damping force of **15000 N·s/m**, and a rotational damping torque of **1700 N·s·m**.

---

# Questions

## Question 1
If a float oscillates vertically in waves, establish its motion equation. Based on the parameters in Appendices 3 and 4, calculate the wave excitation force \( f \cos \omega t \) (where \( f \) is the wave excitation force amplitude and \( \omega \) is the wave frequency) under two scenarios for 40 wave periods with a time interval of 0.2s:

1. The damping coefficient of the direct damper is **10000 N·s/m**.
2. The proportional relationship between the relative velocity of the float and the damper is **10000**, with a noise level of **0.5**.

## Question 2
Considering only vertical oscillation, based on Appendices 3 and 4, calculate the optimal damping coefficient of the direct damper under two scenarios to maximize the average power output of the PTO system:

1. When the damping coefficient is a constant within \([10, 100000]\).
2. When the proportional relationship between the relative velocity of the float and the damper is a constant within \([10, 100000]\) and the noise level is within \([0, 1]\).

## Question 3
For the horizontal-vertical coupled motion in Question 3, calculate the combined work done by the direct and rotational dampers. Provide a detailed analysis of the angular velocity and torque.

## Question 4
Under horizontal-vertical coupling, calculate the maximum power output and optimal damping when the damping coefficients for both the direct and rotational dampers fall within the range \([10, 100000]\).


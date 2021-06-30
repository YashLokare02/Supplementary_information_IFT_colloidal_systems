%% Langevin simulation (without interia) of a particle trapped within a quartic potential well
% Note: Here, an underdamped motion of the particle has been considered (for a relatively small gamma). 

%% Problem setup
% Theoretical background (Ref.:
% https://www.researchgate.net/publication/329413894_Langevin_equation_for_a_particle_in_magnetic_field_is_inconsistent_with_equilibrium)
% Section 1
% The Langevin equation for a particle trapped within a potential V (an unidimensional potential of some form) at a finite temperature T is given by:
% dx/dt = (-1/gamma)*dV/dx + sqrt(2*D)*w(t)......(1)
% Note that: Here, D is the diffusion constant given by: D=k*T/gamma
% (gamma being the friction constant (depends on the radius of the
% particle, its velocity v relative to the fluid and the coefficient of
% viscosity of the fluid)
% Taking this into account, eq.(1) assumes the following form:
% dx/dt = (-1/gamma)*dV/dx + sqrt((2*k*T)/gamma)*w(t)
% Here, w(t) is the Gaussian random noise (characterised by the random
% forces acting on the system)
% Using eq.(1), one can obtain the Langevin equation for a particle trapped
% within a quartic potential well (V(x)=alpha*x^4) as follows:
% dx/dt = -(4*alpha*x^3)/gamma + sqrt((2*k*T)/gamma)*w(t)....(2)
% Finite-difference equation approach for solving the above ODE:
% Note that, dx/dt can be recasted (by first principles) into a more
% conventional form as: dx/dt = (x(i)-x(i-1))/del(t) (where del(t) happens
% to be the time step (initially set) over which the simulation is expected
% to run
% Also, the Gaussian random noise w(t) can be expressed as:
% w(t)=w(i)/sqrt(Dt)....(3), where w(i) happens to be a sequence of Gaussian
% random
% numbers lying between 0 and 1 (zero mean and unit variance)

%% Defining parameters for the problem
N_p= 1;  % Number of particles in the system
kT= 0.772;  % The value of kT is such that it is equal to 0.998
alpha = 0.967;  % Stiffness constant
gamma= 0.1; % Friction constant
D= kT/gamma;  % Einstein diffusion constant
Nt= 4*1.0e+5;  % Number of samples picked/# of iterations
Dt= 1.0e-3;  % Time step for the problem

%% Initialization
x= zeros(N_p, Nt);  % Vector containing the positions of the particle at all times
x(1)= 0; % Particle starts at the origin

%% Numerical computations (using eq(2) and eq(3))

for i=2:Nt
x(i) = x(i-1) - alpha*(x(i-1)*Dt)/gamma;

% Adding a random Gaussian white noise to the system
x(i) = x(i) + sqrt(2*D*Dt)*randn(N_p, 1);  % Note: randn() generates a sequence of random Gaussian numbers ranging from 0 to 1

end

t= (0:Dt:(Nt-1)*Dt);  % Values of t over which the simulation is expected to run

%% Visualization
figure(1);
hold on
plot(t, x, '-bo');
title('Position plot of the particle vs. time', 'FontSize', 15);
xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$x(t)$', 'Interpreter', 'latex', 'FontSize', 16);
hold off

%% Computation of the mean-squared displacements of the particle 
[MSD, t_vec] = LangevinFunction(x, Dt);

figure(2);
grid on
plot(t_vec, MSD, 'b', 'linewidth', 2);
xlabel('t (sampled) [s]', 'FontSize', 22);
ylabel('$\langle(x(t))^2 \rangle$', 'Interpreter', 'latex', 'FontSize', 22, 'Color', 'k');

%% Function to compute the mean-square displacement of the particle

function [MSD,t_array] = LangevinFunction(x, Dt)

% Numerical computations

for n = 0:1:round(sqrt(length(x))) + 100
MSD(n+1) = mean((x(n+1:end)-x(1:end-n)).^2); 
end

t_array = Dt*(0:1:length(MSD)-1);

end











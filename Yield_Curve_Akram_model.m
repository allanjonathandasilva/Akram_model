%% ---------------------------------------------------------------
%  Bond prices and yield curves for the original three-factor model
%  r_t, π_t^E  ~ CIR      |  R_LT driven by (r_t , π_t^E) + own noise
%  Riccati system follows Proposition 1 (b1,b2,c1,c2,a1,a2,a3)
% ---------------------------------------------------------------
close all; clear; clc;

% --------- Parameters (unchanged symbols) -----------------------
b1 = 0.5;         % mean-reversion of r_t
b2 = 0.015;       % volatility of r_t
c1 = 1.0;         % mean-reversion of π_t^E
c2 = 0.5;         % volatility of π_t^E
pibar = 0.01;     % long-run mean of π_t^E

a1 = 0.2;        % loading of dr_t  in dR_LT
a2 = 1 - a1;      % loading of dπ_t^E in dR_LT
a3 = 0.001;        % own volatility of R_LT           ← novo parâmetro

% --------- Initial state (t = 0) -------------------------------
r0   = 0.02;
pi0  = 0.01;
R0   = a1*r0 + a2*pi0;   % valor compatível com definição de R_LT

% --------- Time-to-maturity weight α(τ) ------------------------
lambda = 0.10;
alpha  = @(tau) exp(-lambda*tau);   % α(τ)=e^{-λτ}

% --------- Grid of long-run means for r_t ----------------------
rbar_values = [0.02, 0.03, 0.04];
colors = {'r','k','b'};
labels = {'$\bar{r}=2\%$','$\bar{r}=3\%$','$\bar{r}=4\%$'};

% --------- Maturities -----------------------------------------
Ts   = 0.01:0.2:50;
Pmat = zeros(numel(rbar_values), numel(Ts));

%% ---------- Main loop over rbar -------------------------------
for i = 1:numel(rbar_values)
    rbar = rbar_values(i);          % long-run mean of r_t
    
    for j = 1:numel(Ts)
        T = Ts(j);
        tauSpan = [0 T];            % integrate forward in τ
        y0 = [0 0 0 0];             % [A, B1, B2, B3] at τ=0
        
        opt = odeset('RelTol',1e-7,'AbsTol',1e-9);
        sol = ode45(@(tau,Y) RiccatiODE(tau,Y,alpha,...
                        b1,b2,c1,c2,a1,a2,a3,rbar,pibar),...
                        tauSpan,y0,opt);
        YT = deval(sol,T);          % coefficients at τ = T
        A  = YT(1);  B1 = YT(2);  B2 = YT(3);  B3 = YT(4);
        
        % ------- Bond price at t = 0 ---------------------------
        Pmat(i,j) = exp(-A - B1*r0 - B2*pi0 - B3*R0);
    end
end

%% --------- Plot: bond-price curves ----------------------------
figure; hold on;
for i = 1:numel(rbar_values)
    plot(Ts,Pmat(i,:),colors{i},'LineWidth',2,'DisplayName',labels{i});
end
xlabel('Maturity T (years)'); ylabel('P(0,T)');
title('Zero-coupon bond prices – original model');
legend('Location','best','Interpreter','latex'); grid on;

%% --------- Yield curves ---------------------------------------
Ymat = -log(Pmat)./Ts;    % continuously compounded yield

figure; hold on;
for i = 1:numel(rbar_values)
    plot(Ts,Ymat(i,:),colors{i},'LineWidth',2,'DisplayName',labels{i});
end
xlabel('Maturity T (years)'); ylabel('Yield y(0,T)');
title('Yield curves – original model');
legend('Location','best','Interpreter','latex'); grid on;

%% ==============================================================
%  Nested function: Riccati ODE system dY/dτ  (τ = T-t)
% ==============================================================
function dY = RiccatiODE(tau,Y,alpha,b1,b2,c1,c2,a1,a2,a3,rbar,pibar)

    % unpack state -------------------------------------------------------
    A  = Y(1);  B1 = Y(2);  B2 = Y(3);  
    B3 = Y(4);
    a  = alpha(tau);               % α(τ)
    one_minus_a = 1 - a;

    % --------- Riccati system (stable signs) ----------------------------

    dB1 =  a - b1*B1- a1*b1*B3- 0.5*b2^2*B1^2- 0.5*(a1*b2)^2 *B3^2 - a1*b2^2*B1*B3;

    dB2 = - c1*B2 - a2*c1*B3 - 0.5*c2^2*B2^2- 0.5*(a2*c2)^2*B3^2 - a2*c2^2*B2*B3;

    dB3 =  one_minus_a - 0.5*a3^2*B3^2;

    dA  = b1*rbar*B1 + c1*pibar*B2 + a1*b1*rbar*B3 + a2*c1*pibar*B3;

    dY = [dA; dB1; dB2; dB3];
end


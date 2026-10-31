function summary = run_local_suite(dt, outputName, seeds)
% Two-link planar arm. Requires base MATLAB only; no VLA or MuJoCo.
% Payload is a point mass rigidly attached at the second link endpoint.
if nargin < 1, dt = 0.002; end
if nargin < 2, outputName = 'local_suite'; end
if nargin < 3, seeds = 11:15; end
root = fileparts(fileparts(mfilename('fullpath')));
out = fullfile(root, 'results', outputName);
if ~exist(out,'dir'), mkdir(out); end
p = struct('l1',0.5,'l2',0.4,'m1',2,'m2',1.5, ...
    'g',9.81,'payload',0,'friction',[0.08;0.06]);
p.I1 = p.m1*p.l1^2/12; p.I2 = p.m2*p.l2^2/12;
cfg = struct('dt',dt,'T',8,'Kp',[64;64],'Kd',[16;16], ...
    'Lambda',[8;8],'Ks',[8;3],'rho',[3;2], ...
    'phi',0.2,'limit',[60;40]);
% Common fixed gains and limits for all scenarios. Not Panda parameters.
checks = check_model(p);
cfg.gamma=8; cfg.sigma=0.02; cfg.weight_limit=50;
rng(20261031); cfg.centers=2*rand(6,33)-1; cfg.width=2.5;
scenarios = {'nominal','payload','friction','pulse','combined','unseen_payload','low_torque','measurement_noise'};
controllers = {'PD_gravity','CTC','SMC','RBF_SMC','RBF_NO_ROBUST','RBF_FROZEN'};
records = struct([]); runs = struct(); row = 0;
t = (0:dt:cfg.T)';
for seed=seeds
rng(seed); initial_offset=0.03*randn(2,1);
noise=0.001*randn(numel(t),4);
for si = 1:numel(scenarios)
    actual = p;
    if ismember(si,[2,5,7,8]), actual.payload=0.5; end
    if si==6, actual.payload=0.8; end
    if ismember(si,[3,5]), actual.friction=3*p.friction; end
    localcfg=cfg; if si==7, localcfg.limit=[18;6]; end
    for ci = 1:numel(controllers)
        name = controllers{ci}; scenario = scenarios{si};
        [q0,dq0] = reference(0); x = [q0+initial_offset;dq0]; W=zeros(2,33);
        n = numel(t); X=zeros(n,4); Qd=zeros(n,2);
        Tau=zeros(n,2); Raw=zeros(n,2); Dist=zeros(n,2); Adapt=zeros(n,3);
        for k=1:n
            [qd,~,~] = reference(t(k));
            observed=x; if si==8, observed=observed+noise(k,:)'; end
            [u,raw,Wnew,comp] = control(t(k),observed,p,localcfg,name,W);
            Adapt(k,:)=[norm(W,'fro'),comp'];
            X(k,:)=x'; Qd(k,:)=qd'; Tau(k,:)=u'; Raw(k,:)=raw';
            Dist(k,:)=disturbance(t(k),ismember(si,[4,5]))';
            if k<n
                % Zero-order hold torque and disturbance over one interval.
                d=disturbance(t(k),ismember(si,[4,5])); W=Wnew;
                fun=@(z) plant(z,u,d,actual);
                a=fun(x); b=fun(x+dt*a/2); c=fun(x+dt*b/2); z=fun(x+dt*c);
                x=x+dt*(a+2*b+2*c+z)/6;
                assert(all(isfinite(x)),'Non-finite state');
            end
        end
        e=X(:,1:2)-Qd;
        pos=forward_kinematics(X(:,1:2),p);
        desiredPos=forward_kinematics(Qd,p);
        pe=vecnorm(pos-desiredPos,2,2);
        row=row+1;
        records(row).seed=seed; records(row).scenario=scenario;
        records(row).controller=name;
        records(row).joint_rmse_rad=sqrt(mean(sum(e.^2,2)));
        records(row).ee_rmse_m=sqrt(mean(pe.^2));
        records(row).ee_peak_m=max(pe);
        records(row).torque_rms_Nm=sqrt(mean(sum(Tau.^2,2)));
        records(row).torque_rate_rms_Nm_s=sqrt(mean(sum((diff(Tau)/dt).^2,2)));
        records(row).saturation_fraction=mean(any(abs(Raw)>localcfg.limit',2));
        records(row).max_weight_norm=max(Adapt(:,1));
        records(row).late_ee_rmse_m=sqrt(mean(pe(t>=6).^2));
        data=struct('adaptation',Adapt,'t',t,'state',X,'qd',Qd,'tau',Tau,'raw_tau',Raw, ...
            'disturbance',Dist,'ee_error',pe,'actual',actual,'nominal',p);
        if seed==seeds(1), runs.(scenario).(name)=data; end
        if seed==seeds(1)
        writetable(table(t,X(:,1),X(:,2),X(:,3),X(:,4), ...
            Qd(:,1),Qd(:,2),Tau(:,1),Tau(:,2),Dist(:,1),Dist(:,2),pe, ...
            'VariableNames',{'time_s','q1','q2','dq1','dq2','qd1','qd2', ...
            'tau1','tau2','disturbance1','disturbance2','ee_error_m'}), ...
            fullfile(out,[scenario '_' name '.csv']));
        writematrix([t Adapt],fullfile(out,[scenario '_' name '_adaptation.csv']));
        end
    end
end
end
summary=struct2table(records);
writetable(summary,fullfile(out,'summary.csv'));
metadata=struct('matlab_version',version,'computer',computer, ...
    'generated_at',char(datetime('now')));
save(fullfile(out,'experiment.mat'),'runs','summary','cfg','p','checks','metadata','seeds');
fig=figure('Visible','off','Position',[100 100 1400 1100]);
tiledlayout(4,2);
for si=[1,2,4,5]
    nexttile; hold on;
    for ci=1:numel(controllers)
        r=runs.(scenarios{si}).(controllers{ci});
        plot(r.t,1000*r.ee_error,'LineWidth',1.2);
    end
    title([scenarios{si} ' / endpoint error']); xlabel('Time (s)'); ylabel('Error (mm)'); grid on;
    legend(controllers,'Interpreter','none','Location','best');
    nexttile; hold on;
    for ci=1:numel(controllers)
        r=runs.(scenarios{si}).(controllers{ci});
        plot(r.t,r.tau(:,1),'LineWidth',1.1);
    end
    title([scenarios{si} ' / joint 1 torque']); xlabel('Time (s)'); ylabel('Torque (Nm)'); grid on;
end
exportgraphics(fig,fullfile(out,'comparison.png'),'Resolution',150); close(fig);
disp(summary(1:min(12,height(summary)),:)); disp(checks);
fprintf('Results: %s\n',out);
end

function [M,C,g]=dynamics(q,dq,p)
% Angles measured from horizontal; q2 is relative to link 1.
c1=p.l1/2; c2=p.l2/2; mp=p.payload;
a=p.I1+p.I2+p.m1*c1^2+p.m2*(p.l1^2+c2^2)+mp*(p.l1^2+p.l2^2);
b=p.m2*p.l1*c2+mp*p.l1*p.l2;
d=p.I2+p.m2*c2^2+mp*p.l2^2;
M=[a+2*b*cos(q(2)),d+b*cos(q(2));d+b*cos(q(2)),d];
h=b*sin(q(2));
C=[-h*dq(2),-h*(dq(1)+dq(2));h*dq(1),0];
g=p.g*[(p.m1*c1+(p.m2+mp)*p.l1)*cos(q(1))+(p.m2*c2+mp*p.l2)*cos(sum(q)); ...
    (p.m2*c2+mp*p.l2)*cos(sum(q))];
end

function dx=plant(x,u,d,p)
q=x(1:2); dq=x(3:4); [M,C,g]=dynamics(q,dq,p);
dx=[dq;M\(u+d-C*dq-g-p.friction.*dq)];
end

function [u,raw,Wnext,comp]=control(t,x,p,cfg,name,W)
Wnext=W; comp=zeros(2,1);
q=x(1:2); dq=x(3:4); [qd,dqd,ddqd]=reference(t);
e=q-qd; de=dq-dqd; [M,C,g]=dynamics(q,dq,p);
switch name
    case 'PD_gravity'
        raw=g-cfg.Kp.*e-cfg.Kd.*de;
    case 'CTC'
        raw=M*(ddqd-cfg.Kd.*de-cfg.Kp.*e)+C*dq+g+p.friction.*dq;
    case {'SMC','RBF_SMC','RBF_NO_ROBUST','RBF_FROZEN'}
        s=de+cfg.Lambda.*e;
        vr=dqd-cfg.Lambda.*e; ar=ddqd-cfg.Lambda.*de;
        z=([q;dq;ar]-[0.5;-0.8;0;0;0;0])./[0.5;0.5;1;1;5;5];
        phi=exp(-sum((z-cfg.centers).^2,1)'/(2*cfg.width^2));
        phi=phi/max(norm(phi),1e-10);
        robust=cfg.rho.*max(-1,min(1,s/cfg.phi));
        if ~strcmp(name,'SMC')
            comp=W*phi;
            if ~strcmp(name,'RBF_FROZEN') || t<3
                Wnext=W+cfg.dt*(-cfg.gamma*s*phi'-cfg.sigma*W);
                Wnext=Wnext*min(1,cfg.weight_limit/max(norm(Wnext,'fro'),1e-12));
            end
        end
        if strcmp(name,'RBF_NO_ROBUST'), robust=zeros(2,1); end
        raw=M*ar+C*vr+g+p.friction.*dq-cfg.Ks.*s-robust+comp;
end
u=max(-cfg.limit,min(cfg.limit,raw));
end

function [q,dq,ddq]=reference(t)
w=2*pi/6;
q=[0.5+0.25*sin(w*t);-0.8+0.2*cos(w*t)];
dq=[0.25*w*cos(w*t);-0.2*w*sin(w*t)];
ddq=[-0.25*w^2*sin(w*t);-0.2*w^2*cos(w*t)];
end

function d=disturbance(t,scenario)
d=zeros(2,1);
if scenario && t>=4 && t<4.2, d=[3;-1.5]; end
end

function xy=forward_kinematics(q,p)
xy=[p.l1*cos(q(:,1))+p.l2*cos(sum(q,2)), ...
    p.l1*sin(q(:,1))+p.l2*sin(sum(q,2))];
end

function checks=check_model(p)
minEig=inf; skewError=0; gravityResidual=0;
for mp=[0,0.5,1]
    p.payload=mp;
    for angle=linspace(-pi,pi,31)
        q=[0.4;angle]; v=[0.3;-0.2];
        [M,C,g]=dynamics(q,v,p);
        minEig=min(minEig,min(eig(M)));
        eps=1e-6;
        [Mp,~,~]=dynamics(q+eps*v,v,p); [Mm,~,~]=dynamics(q-eps*v,v,p);
        A=(Mp-Mm)/(2*eps)-2*C;
        skewError=max(skewError,norm(A+A','fro'));
        residual=plant([q;0;0],g,[0;0],p);
        gravityResidual=max(gravityResidual,norm(residual));
    end
end
assert(minEig>0 && skewError<1e-7 && gravityResidual<1e-10,'Dynamics checks failed');
checks=struct('minimum_inertia_eigenvalue',minEig, ...
    'max_skew_identity_residual',skewError,'gravity_equilibrium_residual',gravityResidual);
end

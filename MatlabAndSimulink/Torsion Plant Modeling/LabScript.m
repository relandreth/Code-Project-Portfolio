
J1=0.0103;
J3=0.0099;
k=1.3836;
B1=0.0277;
B3=0.00082;

T=3.1*6; %total time graphed
dt=0.0000125; %Timestep

A=[0,1,0,0;
    -k/J1,-B1/J1,k/J1,0;
    0,0,0,1;
    k/J3,0,-k/J3,-B3/J3];
B=[0;1/J1;0;0];
C=[0,0,0,1];
D=[0];




a=1;
b=[1/8 1/8 1/8 1/8 1/8 1/8 1/8 1/8];
Velocity=diff(position)./diff(eTime);
V2=diff(x3p)./diff(x3t);
FVelocity=filter(b,a,Velocity);
FV2=filter(b,a,V2);


figure('defaultAxesFontSize',14)
subplot(2,1,1)
hold
ylim([-3000,12000])
xlim([0,8])
plot(x3t(1:4496,1),FV2,'k','linewidth',2)
plot(tout,OutputData(1:1488001,2),'k')
xlabel('Time (s)')
ylabel('Velocity(Counts per Second)')
legend('Experimental','Model')
legend('boxoff')
subplot(2,1,2)
hold
ylim([-0.05,0.55])
xlim([0,8])
plot(x3t,x3f,'k','linewidth',2)
xlabel('Time (s)')
ylabel('Input Voltage (V)')
plot(tout,InputData(1:1488001,2),'--k')




%figure('defaultAxesFontSize',14)
%subplot(2,1,1)
%hold
%ylim([-3000,10000])
%xlim([3,6.2])
%plot(eTime(1:10511,1),FVelocity,'k','linewidth',2)
%plot(tout,OutputData(1:1860001,2))
%xlabel('Time (s)')
%ylabel('Velocity(Counts per Second)')
%legend('Experimental','Model')
%legend('boxoff')
%subplot(2,1,2)
%hold
%ylim([-0.05,0.55])
%plot(eTime,sControlEffort,'k','linewidth',2)
%xlabel('Time (s)')
%ylabel('Input Voltage (V)')
%plot(tout,InputData(1:1860001,2))

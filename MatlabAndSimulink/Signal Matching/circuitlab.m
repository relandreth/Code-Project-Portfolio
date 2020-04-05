%Circuit Lab Stuff
R1=4610;
R2=46200;
R3=4580;
C=0.1*10^(-6);
f=2000;
T=1/f;
deltaT=T/1000;

%TimeData2k=time2k(42:1543);
%InputData2k=mag2k(42:1543);
%OutputData2k=mag2k2(42:1543);
%InputData2k=InputData2k-mean(InputData2k);
%OutputData2k=OutputData2k-mean(OutputData2k);
%TimeData2k=TimeData2k-TimeData2k(1);
%Data2k=[TimeData2k,InputData2k,OutputData2k];

%TimeData200=time200(290:1792);
%InputData200=mag200(290:1792);
%OutputData200=mag2002(290:1792);
%InputData200=InputData200-mean(InputData200);
%OutputData200=OutputData200-mean(OutputData200);
%TimeData200=TimeData200-TimeData200(1);
%Data200=[TimeData200,InputData200,OutputData200];

%TimeData100=time100(151:2149);
%InputData100=mag100(151:2149);
%OutputData100=mag1002(151:2149);
%InputData100=InputData100-mean(InputData100);
%OutputData100=OutputData100-mean(OutputData100);
%TimeData100=TimeData100-TimeData100(1);
%Data100=[TimeData100,InputData100,OutputData100];

%TimeDataTri=tritime(103:1605);
%InputDataTri=trimag1(103:1605);
%OutputDataTri=trimag2(103:1605);
%InputDataTri=InputDataTri-mean(InputDataTri);
%utputDataTri=OutputDataTri-mean(OutputDataTri);
%TimeDataTri=TimeDataTri-TimeDataTri(1);
%DataTri=[TimeDataTri,InputDataTri,OutputDataTri];



%figure('defaultAxesFontSize',14)
%subplot(2,1,1)
%hold
%ylim([-2 2])
%xlim([-0.0001 0.0016])
%plot(DataTri(:,1),DataTri(:,2),'k--','linewidth',3)
%plot(simDataTriangle(:,1),simDataTriangle(:,3),'k')
%xlabel('Time (s)')
%ylabel('System Voltage Input (V)')
%legend('Model','Measured')
%legend('boxoff')
%subplot(2,1,2)
%hold
%ylim([-.3 .3])
%lim([-0.0001 0.0016])
%plot(DataTri(:,1),DataTri(:,3),'k--','linewidth',3)
%plot(simDataTriangle(:,1),simDataTriangle(:,2),'k')
%legend('Model','Measured')
%legend('boxoff')
%xlabel('Time (s)')
%label('System Voltage Output (V)')

figure('defaultAxesFontSize',14)
subplot(2,1,1)
hold
ylim([-2 2])
xlim([-0.001 0.022])
plot(Data100(:,1),Data100(:,2),'k--','linewidth',3)
plot(simData100(:,1),simData100(:,3),'k')
xlabel('Time (s)')
ylabel('System Voltage Input (V)')
legend('Model','Measured')
legend('boxoff')
subplot(2,1,2)
hold
ylim([-10 10])
xlim([-0.001 0.022])
plot(Data100(:,1),Data100(:,3),'k--','linewidth',3)
plot(simData100(:,1),simData100(:,2),'k')
legend('Model','Measured')
legend('boxoff')
xlabel('Time (s)')
ylabel('System Voltage Output (V)')



%plot(SimData200(:,1),SimData200(:,2)*-1,SimData200(:,1),SimData200(:,3)*-1)
%plot(Data200(:,1),Data200(:,2),Data200(:,1),Data200(:,3))


%plot(SimData2k(:,1),SimData2k(:,2),SimData2k(:,1),SimData2k(:,3))
%plot(Data2k(:,1),Data2k(:,2),Data2k(:,1),Data2k(:,3))


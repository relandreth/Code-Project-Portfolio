%Created by Roderick Landreth for a Heat Transfer Project, 2018

N = 101; % Number of Nodes, assuing node 1 = 100 deg. C
T1 = 100; %First node temperature
Tinf = 22.72999604; %ambient temp
hAir = 4.7; %Convection Coeff.
kBar = 167; % conduction Coeff.
Ac = 0.00008064; %Cross Section Area
dy = (1/(N-1)); %the change in height between each node
A = 0.0381 * dy; % Perimeter * dy
temp = zeros(1,N);
temp = temp + 50;
temp(1) = T1;

%disp(temp);
%%
for d = 1:30000
    count = 2;
    for i = 2:(N-1)
        temp(count) = ((kBar*Ac*(temp(count-1)+temp(count+1))+(dy*hAir*A*Tinf))/((dy*hAir*A)+(2*kBar*Ac))); 
        count = count + 1;
    end
    temp(N) = ((dy*hAir*(A/2+Ac)*Tinf)+(kBar*Ac*temp(count-1))/((dy*hAir*(A/2+Ac))+(kBar*Ac)));
end
disp(temp);

x = 1:1:N;
y = temp;
plot(x,y)
function dy = A1_2(t, y)
dy=zeros(4,1);
%y(1)=h,y(2)=z,y(3)=h',y(4)=z'
dy(1)=y(3);%h'
dy(2)=y(4);%z'
dy(3)=80000*y(2) - 80000*y(1) - 10000*abs(y(3) - y(4))^(1/2)*(y(3) - y(4))/2433;
%h''
dy(4)=(6250*cos((2801*pi*t)/1000)+(2009*pi*sin((2801*pi*t)/1000))/20-(80000*y(2) - 80000*y(1) - 10000*abs(y(3) - y(4))^(1/2)*(y(3) - y(4)) + 10045*pi*y(2) + 2886708844902639*y(4)/4398046511104) / (1704664960639959/274877906944));
%z''
end


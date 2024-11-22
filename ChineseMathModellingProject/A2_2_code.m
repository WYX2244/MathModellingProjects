p_val=zeros(301*101,3);
for ii=7000:10:10000
    B2=ii;
    for iii=0:1:100
        pow=iii*0.01;
        [tlist,vallist]=ode45(@A2_2,[0:0.1:40],[0,0,0,0]);
        val_col3=vallist(301:1:400,3);
        val_col4=vallist(301:1:400,4);
        P=B2*(val_col3-val_col4).^2;
        SUM=sum(P);
        
        p_val(((ii-7000)/10)*101 + iii+1,1)=B2;
        p_val(((ii-7000)/10)*101 + iii+1,2)=pow;
        p_val(((ii-7000)/10)*101 + iii+1,3)=SUM(1,1)*0.01;
        
    end
end



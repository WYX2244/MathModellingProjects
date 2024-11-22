 p_val_accu=zeros(201,2);
for ii=17000:10:19000
    p_val_accu((ii-17000)/10+1,1)=ii;
    B2=ii;
    [tlist,vallist]=ode45(@A2_1,[0:0.1:50],[0,0,0,0]);
    val_col3=vallist(401:1:500,3);
    val_col4=vallist(401:1:500,4);
    P=B2*((val_col3-val_col4).^2);
    SUM=sum(P);
    p_val_accu((ii-17000)/10+1,2)=SUM(1,1)*0.01;
end

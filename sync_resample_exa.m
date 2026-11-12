function Data=sync_resample_exa(fileName,Fs)

load(fileName);
num_motors=6;

%% FILTERING
[IMU_0,IMU_1,IMU_2]=compare3(IMU_0,IMU_1,IMU_2);
[XKF1_0,XKF1_1]=compare2(XKF1_0,XKF1_1);
[VIBE_0,VIBE_1,VIBE_2]=compare3(VIBE_0,VIBE_1,VIBE_2);

n=size(IMU_0,1);
GYRX=(IMU_0(1:n,4)+IMU_1(1:n,4)+IMU_2(1:n,4))/3;
GYRY=(IMU_0(1:n,5)+IMU_1(1:n,5)+IMU_2(1:n,5))/3;
GYRZ=(IMU_0(1:n,6)+IMU_1(1:n,6)+IMU_2(1:n,6))/3;
ACCX=(IMU_0(1:n,7)+IMU_1(1:n,7)+IMU_2(1:n,7))/3;
ACCY=(IMU_0(1:n,8)+IMU_1(1:n,8)+IMU_2(1:n,8))/3;
ACCZ=(IMU_0(1:n,9)+IMU_1(1:n,9)+IMU_2(1:n,9))/3;
IMU.GYR=[GYRX GYRY GYRZ];
IMU.ACC=[ACCX ACCY ACCZ];
PWM=RCOU(:,11:11+num_motors-1);
for k=1:num_motors
    ESC.("RPM"+num2str(k-1))=eval("ESC_"+num2str(k-1)+"(:,4)");
    ESC.("CURR"+num2str(k-1))=eval("ESC_"+num2str(k-1)+"(:,7)");
end
ATTITUDE.ROLL=ATT(:,4);
ATTITUDE.PITCH=ATT(:,6);
ATTITUDE.YAW=ATT(:,8);
ATTITUDE.DesROLL=ATT(:,3);
ATTITUDE.DesPITCH=ATT(:,5);
ATTITUDE.DesYAW=ATT(:,7);
n=min([size(XKF1_0,1) size(XKF1_1,1)]);
XKF1.ROLL=(unwrap(XKF1_0(1:n,4)/360*2*pi)+unwrap(XKF1_1(1:n,4)/360*2*pi))/2;
XKF1.PITCH=(unwrap(XKF1_0(1:n,5)/360*2*pi)+unwrap(XKF1_1(1:n,5)/360*2*pi))/2;
XKF1.YAW=(unwrap(XKF1_0(1:n,6)/360*2*pi)+unwrap(XKF1_1(1:n,6)/360*2*pi))/2;
XKF1.VN=(XKF1_0(1:n,7)+XKF1_1(1:n,7))/2;
XKF1.VE=(XKF1_0(1:n,8)+XKF1_1(1:n,8))/2;
XKF1.VD=(XKF1_0(1:n,9)+XKF1_1(1:n,9))/2;
n=min([size(VIBE_0,1) size(VIBE_1,1) size(VIBE_2,1)]);
VIBE.ACC=[(VIBE_0(1:n,4)+VIBE_1(1:n,4)+VIBE_2(1:n,4))/3 (VIBE_0(1:n,5)+VIBE_1(1:n,5)+VIBE_2(1:n,5))/3 (VIBE_0(1:n,6)+VIBE_1(1:n,6)+VIBE_2(1:n,6))/3];

%% TIME
Time.IMU=seconds(IMU_0(1:size(IMU.GYR,1),2)/10^6);
Time.PWM=seconds(RCOU(:,2)/10^6);
for k=1:num_motors
    Time.("ESC"+num2str(k-1))=seconds(eval("ESC_"+num2str(k-1)+"(:,2)")/10^6);
end
Time.ATT=seconds(ATT(:,2)/10^6);
Time.XKF1=seconds(XKF1_0(1:size(XKF1.ROLL,1),2)/10^6);
Time.VIBE=seconds(VIBE_0(1:size(VIBE.ACC,1),2)/10^6);

%% BEGIN TIME FROM 0 SECONDS
initTime=min([Time.IMU(1) Time.PWM(1) Time.ESC0(1) Time.ESC1(1) Time.ESC2(1) Time.ESC3(1) Time.ESC4(1) Time.ESC5(1) Time.ATT(1) Time.XKF1(1) Time.VIBE(1)]);
Time.IMU=Time.IMU-initTime;
Time.PWM=Time.PWM-initTime;
for k=1:num_motors
    Time.("ESC"+num2str(k-1))=Time.("ESC"+num2str(k-1))-initTime;
end
Time.ATT=Time.ATT-initTime;
Time.XKF1=Time.XKF1-initTime;
Time.VIBE=Time.VIBE-initTime;

%% LAST TIME
finishTime=max([Time.IMU(end) Time.PWM(end) Time.ESC0(end) Time.ESC1(end) Time.ESC2(end) Time.ESC3(end) Time.ESC4(end) Time.ESC5(end) Time.ATT(end) Time.XKF1(end) Time.VIBE(end)]);
%% TIME
integerUp=3;
timesoutInterp=(0:1/(Fs*integerUp):seconds(finishTime))';

IMU.ACC_sync=resampling(IMU.ACC,seconds(Time.IMU-Time.IMU(1)),timesoutInterp,Fs,integerUp);
IMU.GYR_sync=resampling(IMU.GYR,seconds(Time.IMU-Time.IMU(1)),timesoutInterp,Fs,integerUp);
PWM_sync=resampling(PWM,seconds(Time.PWM-Time.PWM(1)),timesoutInterp,Fs,integerUp);
for k=1:num_motors
    timing=Time.("ESC"+num2str(k-1));
    ESC.("RPM"+num2str(k-1)+"_sync")=resampling(ESC.("RPM"+num2str(k-1)),seconds(timing-timing(1)),timesoutInterp,Fs,integerUp);
    ESC.("CURR"+num2str(k-1)+"_sync")=resampling(ESC.("CURR"+num2str(k-1)),seconds(timing-timing(1)),timesoutInterp,Fs,integerUp);
end
ATTITUDE.ROLL_sync=resampling(ATTITUDE.ROLL,seconds(Time.ATT-Time.ATT(1)),timesoutInterp,Fs,integerUp);
ATTITUDE.PITCH_sync=resampling(ATTITUDE.PITCH,seconds(Time.ATT-Time.ATT(1)),timesoutInterp,Fs,integerUp);
ATTITUDE.YAW_sync=resampling(ATTITUDE.YAW,seconds(Time.ATT-Time.ATT(1)),timesoutInterp,Fs,integerUp);
ATTITUDE.DesROLL_sync=resampling(ATTITUDE.DesROLL,seconds(Time.ATT-Time.ATT(1)),timesoutInterp,Fs,integerUp);
ATTITUDE.DesPITCH_sync=resampling(ATTITUDE.DesPITCH,seconds(Time.ATT-Time.ATT(1)),timesoutInterp,Fs,integerUp);
ATTITUDE.DesYAW_sync=resampling(ATTITUDE.DesYAW,seconds(Time.ATT-Time.ATT(1)),timesoutInterp,Fs,integerUp);
XKF1.ROLL_sync=resampling(XKF1.ROLL,seconds(Time.XKF1-Time.XKF1(1)),timesoutInterp,Fs,integerUp);
XKF1.PITCH_sync=resampling(XKF1.PITCH,seconds(Time.XKF1-Time.XKF1(1)),timesoutInterp,Fs,integerUp);
XKF1.YAW_sync=resampling(XKF1.YAW,seconds(Time.XKF1-Time.XKF1(1)),timesoutInterp,Fs,integerUp);
XKF1.VN_sync=resampling(XKF1.VN,seconds(Time.XKF1-Time.XKF1(1)),timesoutInterp,Fs,integerUp);
XKF1.VE_sync=resampling(XKF1.VE,seconds(Time.XKF1-Time.XKF1(1)),timesoutInterp,Fs,integerUp);
XKF1.VD_sync=resampling(XKF1.VD,seconds(Time.XKF1-Time.XKF1(1)),timesoutInterp,Fs,integerUp);
VIBE.ACC_sync=resampling(VIBE.ACC,seconds(Time.VIBE-Time.VIBE(1)),timesoutInterp,Fs,integerUp);

%% CUT TAKE OFF AND LANDING
idxcut=sum(PWM_sync,2)<1450*size(PWM,2);
IMU.GYR_sync(idxcut,:)=[];
IMU.ACC_sync(idxcut,:)=[];
PWM_sync(idxcut,:)=[];
for k=1:num_motors
    ESC.("RPM"+num2str(k-1)+"_sync")(idxcut,:)=[];
    ESC.("CURR"+num2str(k-1)+"_sync")(idxcut,:)=[];
end
ATTITUDE.ROLL_sync(idxcut,:)=[];
ATTITUDE.PITCH_sync(idxcut,:)=[];
ATTITUDE.YAW_sync(idxcut,:)=[];
ATTITUDE.DesROLL_sync(idxcut,:)=[];
ATTITUDE.DesPITCH_sync(idxcut,:)=[];
ATTITUDE.DesYAW_sync(idxcut,:)=[];
XKF1.ROLL_sync(idxcut,:)=[];
XKF1.PITCH_sync(idxcut,:)=[];
XKF1.YAW_sync(idxcut,:)=[];
XKF1.VN_sync(idxcut,:)=[];
XKF1.VE_sync(idxcut,:)=[];
XKF1.VD_sync(idxcut,:)=[];
VIBE.ACC_sync(idxcut,:)=[];

%% CUT FIRST 2 SECONDS
idxcut=2*Fs;
IMU.GYR_sync(idxcut,:)=[];
IMU.ACC_sync(idxcut,:)=[];
PWM_sync(idxcut,:)=[];
for k=1:num_motors
    ESC.("RPM"+num2str(k-1)+"_sync")(idxcut,:)=[];
    ESC.("CURR"+num2str(k-1)+"_sync")(idxcut,:)=[];
end
ATTITUDE.ROLL_sync(idxcut,:)=[];
ATTITUDE.PITCH_sync(idxcut,:)=[];
ATTITUDE.YAW_sync(idxcut,:)=[];
ATTITUDE.DesROLL_sync(idxcut,:)=[];
ATTITUDE.DesPITCH_sync(idxcut,:)=[];
ATTITUDE.DesYAW_sync(idxcut,:)=[];
XKF1.ROLL_sync(idxcut,:)=[];
XKF1.PITCH_sync(idxcut,:)=[];
XKF1.YAW_sync(idxcut,:)=[];
XKF1.VN_sync(idxcut,:)=[];
XKF1.VE_sync(idxcut,:)=[];
XKF1.VD_sync(idxcut,:)=[];
VIBE.ACC_sync(idxcut,:)=[];

%% CUT LAST 2 SECONDS
IMU.GYR_sync(end-idxcut:end,:)=[];
IMU.ACC_sync(end-idxcut:end,:)=[];
PWM_sync(end-idxcut:end,:)=[];
for k=1:num_motors
    ESC.("RPM"+num2str(k-1)+"_sync")(end-idxcut:end,:)=[];
    ESC.("CURR"+num2str(k-1)+"_sync")(end-idxcut:end,:)=[];
end
ATTITUDE.ROLL_sync(end-idxcut:end,:)=[];
ATTITUDE.PITCH_sync(end-idxcut:end,:)=[];
ATTITUDE.YAW_sync(end-idxcut:end,:)=[];
ATTITUDE.DesROLL_sync(end-idxcut:end,:)=[];
ATTITUDE.DesPITCH_sync(end-idxcut:end,:)=[];
ATTITUDE.DesYAW_sync(end-idxcut:end,:)=[];
XKF1.ROLL_sync(end-idxcut:end,:)=[];
XKF1.PITCH_sync(end-idxcut:end,:)=[];
XKF1.YAW_sync(end-idxcut:end,:)=[];
XKF1.VN_sync(end-idxcut:end,:)=[];
XKF1.VE_sync(end-idxcut:end,:)=[];
XKF1.VD_sync(end-idxcut:end,:)=[];
VIBE.ACC_sync(end-idxcut:end,:)=[];

%% SAVE
Data.IMU.GYR=IMU.GYR;
Data.IMU.ACC=IMU.ACC;
Data.PWM=PWM;
for k=1:num_motors
    Data.ESC.("RPM"+num2str(k-1))=ESC.("RPM"+num2str(k-1));
    Data.ESC.("CURR"+num2str(k-1))=ESC.("CURR"+num2str(k-1));
end
Data.ATTITUDE.ROLL=ATTITUDE.ROLL;
Data.ATTITUDE.PITCH=ATTITUDE.PITCH;
Data.ATTITUDE.YAW=ATTITUDE.YAW;
Data.ATTITUDE.DesROLL=ATTITUDE.DesROLL;
Data.ATTITUDE.DesPITCH=ATTITUDE.DesPITCH;
Data.ATTITUDE.DesYAW=ATTITUDE.DesYAW;
Data.XKF1.ROLL=XKF1.ROLL;
Data.XKF1.PITCH=XKF1.PITCH;
Data.XKF1.YAW=XKF1.YAW;
Data.XKF1.VN=XKF1.VN;
Data.XKF1.VE=XKF1.VE;
Data.XKF1.VD=XKF1.VD;
Data.VIBE.ACC=VIBE.ACC;

Data.IMU.GYR_sync=IMU.GYR_sync;
Data.IMU.ACC_sync=IMU.ACC_sync;
Data.PWM_sync=PWM_sync;
for k=1:num_motors
    Data.ESC.("RPM"+num2str(k-1)+"_sync")=ESC.("RPM"+num2str(k-1)+"_sync");
    Data.ESC.("CURR"+num2str(k-1)+"_sync")=ESC.("CURR"+num2str(k-1)+"_sync");
end
Data.ATTITUDE.ROLL_sync=ATTITUDE.ROLL_sync;
Data.ATTITUDE.PITCH_sync=ATTITUDE.PITCH_sync;
Data.ATTITUDE.YAW_sync=ATTITUDE.YAW_sync;
Data.ATTITUDE.DesROLL_sync=ATTITUDE.DesROLL_sync;
Data.ATTITUDE.DesPITCH_sync=ATTITUDE.DesPITCH_sync;
Data.ATTITUDE.DesYAW_sync=ATTITUDE.DesYAW_sync;
Data.XKF1.ROLL_sync=XKF1.ROLL_sync;
Data.XKF1.PITCH_sync=XKF1.PITCH_sync;
Data.XKF1.YAW_sync=XKF1.YAW_sync;
Data.XKF1.VN_sync=XKF1.VN_sync;
Data.XKF1.VE_sync=XKF1.VE_sync;
Data.XKF1.VD_sync=XKF1.VD_sync;
Data.VIBE.ACC_sync=VIBE.ACC_sync;
clear
clc
%Global variable-----------------------------------------------------------
x = 1000;
y = 500;
global obj;
global connector;
global mode_state;
connector = struct('a',[],'b',[]);
mode_state = 0;
obj = struct('figure',[],...
    'BLE_list',[],'Connect',[],'Disconnect', [],'Scan',[],...
    'd_joint',[],'a_joint',[],...
    'Roll',[],'Pitch',[],'Yaw',[],...
    'Auto_list',[],'Run',[],'Pause',[],'Stop',[]);
obj.a_joint = cell(1,12);
obj.d_joint = cell(1,12);
%Gui----------------------------------------------------------------------- 
obj.figure = uifigure('Position', [500 200 500 400]);
    %Graph creating
for i = 1:4
    for j = 1:3
        obj.d_joint{j+(i-1)*3} = uislider(obj.figure,'Position',[18+(i-1)*120 160+80*(j-1) 100 22], 'Limits',[-90 90],'Value',0);
        obj.a_joint{j+(i-1)*3} = uieditfield(obj.figure,'numeric','Position',[68+(i-1)*120 180+80*(j-1) 50 22],'Value',j+(i-1)*3);
        label = uilabel(obj.figure,'Position',[18+(i-1)*120 180+80*(j-1) 50 22],'Text',strcat('Joint_',num2str(i),'.',num2str(j)));
    end
end
obj.BLE_list = uidropdown(obj.figure,'Position',[18 90 100 22],'Items',{});
obj.Scan = uibutton(obj.figure,'Position',[126 90 70 22],'Text','Scan','ButtonPushedFcn',@Scan_qua);
obj.Connect = uibutton(obj.figure,'Position',[200 90 70 22],'Text','Connect','ButtonPushedFcn',@Con_qua);
obj.Disconnect = uibutton(obj.figure,'Position',[274 90 70 22],'Text','Disconnect');
obj.Mode_list = uidropdown(obj.figure,'Position',[18 60 100 22],'Items',{'Manual';'Trot'});
obj.Run = uibutton(obj.figure,'Position',[126 60 70 22],'Text','Run','ButtonPushedFcn',@Run_qua,'Interruptible','on');
obj.Pause = uibutton(obj.figure,'state','Position',[200 60 70 22],'Text','Pause','Enable','off','ValueChangedFcn',@Pause_qua);
obj.Stop = uibutton(obj.figure,'Position',[274 60 70 22],'Text','Stop','Enable','off','ButtonPushedFcn',@Stop_qua);
uilabel(obj.figure,'Position',[360 90 50 22],'Text','Roll');
obj.Roll = uieditfield(obj.figure,'numeric','Position',[420 90 50 22]);
uilabel(obj.figure,'Position',[360 60 50 22],'Text','Pitch');
obj.Pitch = uieditfield(obj.figure,'numeric','Position',[420 60 50 22]);
uilabel(obj.figure,'Position',[360 30 50 22],'Text','Yaw');
obj.Yaw = uieditfield(obj.figure,'numeric','Position',[420 30 50 22]);

%Callback------------------------------------------------------------------
function Scan_qua(src,even)
    global obj;
    list = blelist;
    for i = 1:size(list,1)
        if strcmp(list.Name(i),"Quadrup")
            obj.BLE_list.Items = "Quadrup";
            obj.BLE_list.Value = "Quadrup";
            break;
        end
    end
end
%--------------------------------------------------------------------------
function Con_qua(src,even)
    global obj;
    global connector;
    try
        connector.a = ble(obj.BLE_list.Value);
        connector.c = characteristic(connector.a,"a15b89c6-1042-4c05-af06-52bb41e51c1e","a15b89c6-1042-4c05-af06-52bb41e51c1e");
        disp("ok");
    catch error
        connector.a = [];
        connector.c = [];
        disp(error);
    end
end
%--------------------------------------------------------------------------
function Run_qua(src,even)
    global mode_state;
    global obj;
    mode_state = 1;
    obj.Run.Enable = 'off';
    obj.Pause.Enable = 'on';
    obj.Stop.Enable = 'on';
    pause(5);
    action();
end
%--------------------------------------------------------------------------
function Pause_qua(src,even)
    global mode_state;
    global obj;
    if (mode_state == 1)||(mode_state==2)
        if obj.Pause.Value == 1
            mode_state = 2;
        else 
            mode_state = 1;
        end
    end
end
%--------------------------------------------------------------------------
function Stop_qua(src,even)
    global mode_state;
    global obj;
    mode_state = 0;
    obj.Run.Enable = 'on';
    obj.Pause.Enable = 'off';
    obj.Pause.Value = 1;
    obj.Stop.Enable = 'off';
    pause(5)
end
%Support function----------------------------------------------------------
function action()
    global obj;
    switch obj.Mode_list.Value
        case 'Manual'
            Manual();
    end
end
%--------------------------------------------------------------------------
function disp_value(rdata)
    global obj;
    obj.Roll.Value = rdata(1);
    obj.Pitch.Value = rdata(2);
    obj.Yaw.Value = rdata(3);
    for i = 1:4
        for j = 1:3
            obj.a_joint{j+(i-1)*3}.Value = rdata(j+(i-1)*3+3);
        end
    end
end
%--------------------------------------------------------------------------
function value = take_value()
    global obj;
    value = zeros(1,12);
    for i = 1:4
        for j = 1:3
            value(j+(i-1)*3) = obj.d_joint{j+(i-1)*3}.Value;
        end
    end
end
%--------------------------------------------------------------------------
function set_value(rdata)
    global obj;
    for i = 1:4
        for j = 1:3
            obj.d_joint{j+(i-1)*3}.Value = rdata(j+(i-1)*3);
        end
    end
end
%--------------------------------------------------------------------------
function val = read_data()
    global connector;
    data = read(connector.c);
    val = zeros(1,15);
    if (data(1) == data(38)) && (length(data) == 38)    
        for i = 0:2
            val(i+1) = float_convert(data(4*i+2:4*i+5));    
        end
        for i = 3:14
            val(i+1) = int_convert(data(2*i+7:2*i+10));
        end
    end
end
%--------------------------------------------------------------------------
function send_data(angle_send)
    global connector;
    val = zeros(1,12);
    for i = 1:12
        val(2*i) = (uint8(bitshift(int16(angle_send(i)+90),-8)));
        val(2*i-1) = (uint8(bitand(int16(angle_send(i)+90),255)));
    end
    write(connector.c,char(val));
end
%--------------------------------------------------------------------------
function val = float_convert(byte)
    b = uint8([byte(4),byte(3),byte(2),byte(1)]);
    val = typecast(b, 'single');
end
%--------------------------------------------------------------------------
function val = int_convert(byte)
    b = uint8([byte(2),byte(1)]);
    val = typecast(b, 'int16');
end
%Manual--------------------------------------------------------------------
function Manual()
    global obj;
    global mode_state;
    pre_d_joint = zeros(1,12);
    obj.Pause.Enable = 'off';
    %set_value([0,0,0,0,0,0,0,0,0,0,0,0]);
    while mode_state == 1
        d_joint = take_value();
        if  ~isequaln(pre_d_joint,d_joint)
            send_data(d_joint);
            pre_d_joint = d_joint;
        end
        disp(d_joint);
        a_joint = read_data();
        disp_value(a_joint);
        drawnow;
    end
    obj.Pause.Enable = 'on';
end



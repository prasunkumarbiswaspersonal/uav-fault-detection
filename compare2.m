function [in1,in2] = compare2(in1,in2)
delete=find(~ismember(in1(:,2),in2(:,2)));
in1(delete,:)=[];
delete=find(~ismember(in2(:,2),in1(:,2)));
in2(delete,:)=[];
if(~(isequal(in1(:,2),in2(:,2))))
    error("Inputs to be averaged do not share the same timestamps.")
end
end
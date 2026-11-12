function [in1,in2,in3] = compare3(in1,in2,in3)
delete=find(~ismember(in1(:,2),in2(:,2)) | ~ismember(in1(:,2),in3(:,2)));
in1(delete,:)=[];
delete=find(~ismember(in2(:,2),in1(:,2)) | ~ismember(in2(:,2),in3(:,2)));
in2(delete,:)=[];
delete=find(~ismember(in3(:,2),in1(:,2)) | ~ismember(in3(:,2),in2(:,2)));
in3(delete,:)=[];
if(~(isequal(in1(:,2),in2(:,2)) & isequal(in1(:,2),in3(:,2))))
    error("Inputs to be averaged do not share the same timestamps.")
end
end
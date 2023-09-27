function [D_index, R_index] = SVDD(X,Xt)
    global Cp
    
    Cp = 0.1
    sigma = 50

    % Cp是惩罚因子，较小的Cp会导致更多的支持向量，较大的Cp会导致更少的支持向量。
    % sigma是高斯核函数的参数,较小的sigma会导致样本更密集地聚集在球形区域内，较大的值会使得聚集区域更扩散。

    % 过程监测中, Cp越大, 检测出的异常点越多
    % 过程监测中, sigma越大, 检测出的异常点越少, 但是异常点的检测能力越强, 也就是说, 越容易检测出异常点.

    X = X;
    Y = Xt;
    
    [n,m]=size(X);
    s = std(X);
    mm = mean(X);
    X = (X-ones(n,1)*mm)*diag((s.^(-1)));
    sample_number = size(Y,1);
    Y = (Y-ones(sample_number,1)*mm)*diag((s.^(-1)));
    
    TrainMatrix = X;
    TestMatrix = Y;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%train&test%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [Rows,Columns] = size(TrainMatrix); 
    KM = zeros(Rows,Rows);
    for i=1:Rows
        for j=1:Rows
            s = TrainMatrix(i,:) - TrainMatrix(j,:);
            KM(i,j) = exp(-(norm(s)^2)./(sigma^2));
        end
    end
    [alpha R] = svdd_solve(KM);
    SV = TrainMatrix(alpha>0,:);
    out = predict(KM,TestMatrix,SV,alpha,sigma);
    SupportVector = find(alpha>0);

    D_index = out;
    R_index = R;
end

function [alpha R] = svdd_solve(in_KM)
global Cp;
L = size(in_KM,1);
alpha= zeros(1,L);
alpha(3) = 1;
G = zeros(1,L);

for i=1:L
    if alpha(i)>0
        G(:) = G(:) + alpha(i) * in_KM(:,i);
    end
end

while(1)
    [i j b_exit] = svdd_select_sets(alpha,G,L);
    if b_exit
        break;
    end
    
    old_alpha_i = alpha(i);
    old_alpha_j = alpha(j);
    
    delta = (G(i)-G(j))/max(in_KM(i,i) + in_KM(j,j)-2 * in_KM(i,j),0);
    sum = alpha(i) + alpha(j);
    
    alpha(j) = alpha(j) + delta;
    alpha(i) = alpha(i) - delta;
    
    
    if (alpha(i)<0)
        alpha(i) = 0;
        alpha(j) = sum;
    end
    if (alpha(j)<0)
        alpha(j) = 0;
        alpha(i) = sum;
    end
       %CpΪĬ�ϵĳͷ�����1
    if (alpha(i)>Cp)
        alpha(i) = Cp;
        alpha(j) = sum - Cp;
    end
    if (alpha(j)>Cp)
        alpha(j) = Cp;
        alpha(i) = sum - Cp;
    end
        
    delta_alpha_i = alpha(i) - old_alpha_i;
    delta_alpha_j = alpha(j) - old_alpha_j;
    
    G(:) = G(:) + in_KM(:,i) * delta_alpha_i + in_KM(:,j) * delta_alpha_j;
end %����whileѭ��




%���޸�һ�³��򲿷�
R=0;
for i=1:L
    if (alpha(i)>0)
        for j=1:L
            if (alpha(j)>0)
                R = R + alpha(i) * alpha(j) * in_KM(i,j);
            end
        end
    end
end

all = 0;
times = 0;
for i=1:L
    if ((alpha(i)>0) &&(alpha(i)<Cp))
        tt = 0;
        for j=1:L
            tt = tt - 2 * alpha(j) * in_KM(i,j);
        end
        tt = tt + R + 1;
        all = all + sqrt(tt);
        times = times + 1;
    end
end
R = all / times;
end

function [s,t,b_exit] = svdd_select_sets(alpha,G,L)
global Cp;
Gmax1 = -Inf;
Gmax1_idx = -1;
Gmax2 = -Inf;
Gmax2_idx = -1;

eps = 1e-5;
for i=1:L
    if (alpha(i)<Cp)
        if (-G(i)>Gmax1+1e-15)
            Gmax1 = -G(i);
            Gmax1_idx = i;
        end
    end
    if (alpha(i)>0)
        if (G(i)>(Gmax2+1e-15))
            Gmax2 = G(i);
            Gmax2_idx = i;
        end
    end
end

    s = Gmax1_idx;
    t = Gmax2_idx;



if((Gmax1+Gmax2) < (0.5*eps))
    b_exit = 1;
else
    b_exit = 0;
end

end

function out = predict(KM,testMatrix,SV,alpha,kesi)
L = size(alpha,2);

TM = [testMatrix' SV']';
TM = KernelMatrix(TM,kesi);

[tR tC] = size(testMatrix);
[sR sC] = size(SV);

alph_i = zeros(1,sR);

sub1 = 0;
ii = 0;
for i=1:L
    if (alpha(i)>0)
        ii = ii+1;
        alph_i(ii) = alpha(i);
    end
    
    for j=1:L
        if ((alpha(i)>0)&&(alpha(j)>0))
            sub1 = sub1 + alpha(i) * alpha(j) * KM(i,j);
        end
    end
end

out = zeros(1,tR);
for i=1:tR
    sub2 = 0;
    for j=1:sR
        sub2 = sub2 + alph_i(j) * TM(i,tR+j);
    end
    sub2 = sqrt(1 -2 * sub2 + sub1);
    out(i) = sub2;
end

end

function [KM] = KernelMatrix(TrainMatrix,Sigma)
[Rows,Columns] = size(TrainMatrix); 
KM = zeros(Rows,Rows);
for i=1:Rows
    for j=1:Rows
        s = TrainMatrix(i,:) - TrainMatrix(j,:);
        t = norm(s);
        KM(i,j) = exp(-(t^2)/(Sigma^2));
    end
end
    
end
        



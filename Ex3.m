close all
clear all
clc

%% Q1 - 2D Fourier Transform
%% 1.1
%% 1.1.1
% Write our function 2D-FFT and inverse FFT

%% 1.1.2
% Write our function FFT shift 

%% 1.1.3
beatle = imread('beatles.png');
beatle_gray = rgb2gray(beatle);
beatle_gray_norm = normal(beatle_gray);
figure('Name','1.1.3')
imshow(beatle_gray_norm)
title('Gray scale of the image - Beatles')

%% 1.1.4
beatle_fft = dip_fft2(beatle_gray_norm);
beatle_fft_shift = dip_fftshift(beatle_fft);
beatle_fft_shift_angle = angle(beatle_fft_shift);
beatle_fft_shift_amp = abs(beatle_fft_shift);

figure('Name','1.1.4')
subplot(2,1,1);
imagesc(log(beatle_fft_shift_amp));
colorbar;
title('Log Amplitude')
subplot(2,1,2);
imagesc(beatle_fft_shift_angle);
colorbar;
title('Phase')

% Comparison to MATLAB functions
beatle_fft_MAT = fft2(beatle_gray_norm);
beatle_fft_shift_MAT = fftshift(beatle_fft_MAT);
beatle_fft_shift_angle_MAT = angle(beatle_fft_shift_MAT);
beatle_fft_shift_amp_MAT = abs(beatle_fft_shift_MAT);

figure('Name','1.1.4 - MATLAB Function')
sgtitle('Using MATLAB Functions')
subplot(2,1,1);
imagesc(log(beatle_fft_shift_amp_MAT));
colorbar;
title('Log Amplitude')
subplot(2,1,2);
imagesc(beatle_fft_shift_angle_MAT);
colorbar;
title('Phase')

ssim_amp_FFT2D = ssim(beatle_fft_shift_amp,beatle_fft_shift_amp_MAT);
mse_amp_FFT2D =  immse(beatle_fft_shift_amp,beatle_fft_shift_amp_MAT);

ssim_phase_FFT2D = ssim(beatle_fft_shift_angle,beatle_fft_shift_angle_MAT);
mse_phase_FFT2D =  immse(beatle_fft_shift_angle,beatle_fft_shift_angle_MAT);

%% 1.1.5
beatle_recon = dip_ifft2(beatle_fft);
beatle_recon_MAT = ifft2(beatle_fft_MAT);
figure("Name",'1.1.5');
subplot(3,1,1);
imshow(beatle_gray_norm)
title('Original image - gray scale')
subplot(3,1,2);
imshow(real(beatle_recon))
title('Reconstrunct image')
subplot(3,1,3);
imshow(beatle_recon_MAT)
title('Reconstrunct image - using MATLAB function')

% Similarity between the reconsructions
ssim_recon = ssim(beatle_gray_norm,real(beatle_recon));
mse_recon =  immse(beatle_gray_norm,real(beatle_recon));

ssim_recon_func = ssim(beatle_recon_MAT,real(beatle_recon));
mse_recon_func =  immse(beatle_recon_MAT,real(beatle_recon));
%% 1.2.1
%% 1.2.1.a
file = load('freewilly.mat');
free_willy = file.freewilly;
figure('Name','1.2.1.a')
imshow(free_willy);
title('Free Willy image')

%% 1.2.1.b
figure("Name",'1.2.1.b - Sine in time')
plot(free_willy(1,:))
title('The first row of the image')
xlabel('x')
ylabel('first row')
N = size(free_willy,2);
fft_freewilly = fftshift(fft(free_willy(1,:)));
single_side_fft = fft_freewilly(N/2+2:end);
fVals=(1:N/2-1)/N;
figure("Name",'1.2.1.b - DFT sine')
plot(fVals,abs(single_side_fft))
title('Single Sided FFT - with FFTShift');       
xlabel('Frequency')       
ylabel('DFT Values');

[Y,I] = max(abs(single_side_fft));
fx = abs(fVals(I))*N;

%% 1.2.1.c
x=0:N-1;
M = size(free_willy,1);
bars = 0.5 * sin(2*pi*fx.*x/N);
prison_bars=repmat(bars,M,1);
figure("Name",'1.2.1.c')
imshow(prison_bars)
title('Prison Bars')
colorbar;

%% 1.2.1.d
prison_bars_FFT2 = fftshift(fft2(prison_bars));
figure("Name",'1.2.1.d')
imshow(abs(prison_bars_FFT2))
title('2D FFT of Prison Bars')
colormap('default')
colorbar;

%% 1.2.1.e
willy_without_bars = Free_Willy(free_willy);

%% 1.2.2.a
matrix1_128x128 = zeros(128,128);
matrix1_128x128(44:83,44:83) = 1;
matrix1_128x128_fft2D = fft2(matrix1_128x128);
figure('Name','1.1.2.a');
imagesc(matrix1_128x128)
title('Matrix_{128x128}')
colorbar;

figure('Name','1.1.2.a - Phase and Amplitude');
subplot(1,2,1);
imagesc(angle(fftshift(matrix1_128x128_fft2D)))
title('Matrix_{128x128} - Phase of FFT 2D','FontSize',25)
colorbar;
subplot(1,2,2);
imagesc(log(abs(fftshift(matrix1_128x128_fft2D))))
title('Matrix_{128x128} - Amplitude of FFT 2D','FontSize',25)
colorbar;

% Amplitude of FFT 2D of the image
figure("Name",'Amplitude of FFT 2D of the image');
surf(abs(fftshift(matrix1_128x128_fft2D)));
sgtitle('Amplitude of FFT 2D of the image');
xlabel('\omega_x');
ylabel('\omega_y'); 
zlabel('Amplitude');
colormap("cool")
colorbar;

%% 1.2.2.b
matrix2_128x128 = zeros(128,128);
matrix2_128x128(64:103,64:103) = 1;
matrix2_128x128_fft2D = fft2(matrix2_128x128);
figure('Name','1.1.2.b');
imagesc(matrix2_128x128)
title('Matrix_{128x128}(64:103,64:103)')
colorbar;

figure('Name','1.1.2.b - Phase and Amplitude');
subplot(1,2,1);
imagesc(angle(fftshift(matrix2_128x128_fft2D)))
title('Matrix_{128x128}(64:103,64:103) - Phase of FFT 2D','FontSize',25)
colorbar;
subplot(1,2,2);
imagesc(log(abs(fftshift(matrix2_128x128_fft2D))))
title('Matrix_{128x128}(64:103,64:103) - Amplitude of FFT 2D','FontSize',25)
% colormap("cool")
colorbar;

% Comperison betwwn the Amplitude ao section (a) and (b)
figure('Name','1.1.2.b - Amplitudes of (1.1.2.a , 1.1.2.b)');
sgtitle('Amplitude FFT 2D of section:','FontSize',25)
subplot(1,2,1);
imagesc(log(abs(fftshift(matrix1_128x128_fft2D))))
title('Matrix_{128x128}(44:83,44:83) - 1.1.2.a','FontSize',25)
colorbar;
subplot(1,2,2);
imagesc(log(abs(fftshift(matrix2_128x128_fft2D))))
title('Matrix_{128x128}(64:103,64:103) - 1.1.2.b','FontSize',25)
colorbar;

% Matematical comparation between the Amplitude
ssim_amp = ssim(abs(matrix1_128x128_fft2D),abs(matrix2_128x128_fft2D));
mse_amp =  immse(abs(matrix1_128x128_fft2D),abs(matrix2_128x128_fft2D));

ssim_phase = ssim(angle(fftshift(matrix1_128x128_fft2D)),angle(fftshift(matrix2_128x128_fft2D)));
mse_phase =  immse(angle(fftshift(matrix1_128x128_fft2D)),angle(fftshift(matrix2_128x128_fft2D)));

%% 1.2.2.c
matrix3_128x128 = zeros(128,128);
matrix3_128x128(24:103,54:73) = 1;
matrix3_128x128_fft2D = fft2(matrix3_128x128);
figure('Name','1.1.2.a');
imagesc(matrix3_128x128)
title('Matrix_{128x128}(80X20)')
colorbar;

figure('Name','1.1.2.a - Phase and Amplitude');
subplot(1,2,1);
imagesc(angle(fftshift(matrix3_128x128_fft2D)))
title('Matrix_{128x128}(80X20) - Phase of FFT 2D','FontSize',25)
colorbar;
subplot(1,2,2);
imagesc(log(abs(fftshift(matrix3_128x128_fft2D))))
title('Matrix_{128x128}(80X20) - Amplitude of FFT 2D','FontSize',25)
colorbar;

% Matematical comparation between section (a) and (c)
% Between images:
ssim_images = ssim(matrix1_128x128,matrix3_128x128);
mse_images =  immse(matrix1_128x128,matrix3_128x128);

% Between FFT 2D images:
ssim_FFT2D_images = ssim(real(matrix1_128x128_fft2D),real(matrix3_128x128_fft2D));
mse_FFT2D_images =  immse(matrix1_128x128_fft2D,matrix3_128x128_fft2D);

%% 1.2.2.d
box_vector1 = zeros(1,128);
box_vector1(24:103) = 1;
box_vector2 = zeros(1,128);
box_vector2(54:73) = 1 ;

%% 1.2.2.e
F_1D = sep_fft2(box_vector1,box_vector2);
figure("Name",'1.2.2.e');
sgtitle('Calculate image 2D FFT by:')
subplot(2,2,1);
imagesc(log(abs(F_1D)))
title('Amplitude - Using 1D sst')
subplot(2,2,2);
imagesc(log(abs(fftshift(matrix3_128x128_fft2D))))
title('Amplitude - Using 2D sst')
subplot(2,2,3);
imagesc(angle(F_1D))
title('Phase - Using 1D sst')
subplot(2,2,4);
imagesc(angle(fftshift(matrix3_128x128_fft2D)))
title('Phase - Using 2D sst')

% Matematical comparation between the result

ssim_FFT2D_1D = ssim(real(F_1D),real(fftshift(matrix3_128x128_fft2D)));
mse_FFT2D_1D = immse(F_1D,fftshift(matrix3_128x128_fft2D));

%% Q2 - Discrete Cosine Transform
%% 2.1
beatle = imread('beatles.png');
beatle_gray = rgb2gray(beatle);
beatle_gray_norm = normal(beatle_gray);

%% 2.2
beatle_DCT = dct2(beatle_gray_norm);
figure('Name','2.2')
imagesc(log(abs(beatle_DCT)))
sgtitle('Logarithmic display of Beatles DCT')
colormap(jet(64))
colorbar;

%% 2.3
% Randomly setting 50% of the DCT values to 0
Idx = randperm(numel(beatle_DCT),ceil(numel(beatle_DCT) * 0.50));
beatle_DCT_50_pre = beatle_DCT;
beatle_DCT_50_pre(Idx) = 0;
beatle_IDCT_50_pre = idct2(beatle_DCT_50_pre);
figure("Name",'2.3')
imshow(beatle_IDCT_50_pre)
title('IDCT of the image after setting 50% of the DCT values to 0')

% Matematical comparation between the reconstruct

ssim_rand = ssim(beatle_IDCT_50_pre,beatle_gray_norm);
mse_rand = immse(beatle_IDCT_50_pre,beatle_gray_norm);

%% 2.4
% Setting the 50% absolute lowest values of the DCT to 0
beatle_DCT_lowest_50_pre = beatle_DCT;
[mi_abs_val,index] = mink(abs(reshape(beatle_DCT_lowest_50_pre,1,[])),floor(numel(beatle_DCT_lowest_50_pre)*0.5));
beatle_DCT_lowest_50_pre(index) = 0;
beatle_IDCT_lowest_50_pre = idct2(beatle_DCT_lowest_50_pre);
figure("Name",'2.4')
imshow(beatle_IDCT_lowest_50_pre)
title('IDCT of the image after setting the 50% absolute lowest values of the DCT to 0')

% Matematical comparation between the reconstruct

ssim_lowest = ssim(beatle_IDCT_lowest_50_pre,beatle_gray_norm);
mse_lowest = immse(beatle_IDCT_lowest_50_pre,beatle_gray_norm);

%% 2.5
% Setting the DCT values in range (-a,a) to 0
% beatle_DCT_range_a = beatle_DCT;
figure('Name','2.5 - [0 - 0.015]');
sgtitle('IDCT of the image for zero range of:')
val2zero(beatle_DCT,0,0.005);

figure('Name','2.5 - [0.3 - 1.8]');
sgtitle('IDCT of the image for zero range of:')
val2zero(beatle_DCT,0.3,0.5);

figure('Name','2.5 - [2 - 2.75]');
sgtitle('IDCT of the image for zero range of:')
val2zero(beatle_DCT,2,0.25);

figure('Name','2.5 - [3 - 9]');
sgtitle('IDCT of the image for zero range of:')
val2zero(beatle_DCT,3,2);

% The precent of DCT values that set to 0 for a = 0.015

pre_a = val2zero(beatle_DCT,0,0.005);
disp(pre_a);

%% Q3 - Wavelet Transform
%% 3.1
BEETLE = imread('beetle.jpg');
BEETLE_gray = rgb2gray(BEETLE);
BEETLE_gray_norm = normal(BEETLE_gray);
figure('Name','beetle');
imshow(BEETLE_gray_norm);
title('Gray scale of the image - BEETLE');

%% 3.2
[C3,S3] = wavedec2(BEETLE_gray_norm,3,'haar');
[C4,S4] = wavedec2(BEETLE_gray_norm,4,'haar');
[C5,S5] = wavedec2(BEETLE_gray_norm,5,'haar');
figure('Name','3.2');
sgtitle('Decomposition of:')
subplot(3,1,1);
plot(C3)
title('level 3')
subplot(3,1,2);
plot(C4)
title('level 4')
subplot(3,1,3);
plot(C5)
title('level 5')

%% 3.3
%H - horizontal, V - vertical, D - diagonal
%Finding the detail and approximation coefficients
[H3,V3,D3] = detcoef2('all',C3,S3,3);
approx3 = appcoef2(C3,S3,'Haar',3);
[H4,V4,D4] = detcoef2('all',C4,S4,4);
approx4 = appcoef2(C4,S4,'Haar',4);
[H5,V5,D5] = detcoef2('all',C5,S5,5);
approx5 = appcoef2(C5,S5,'Haar',5);

%% 3.4
%Level 3:
figure("Name",'3.4 - LEVEL 3');
sgtitle('Detail & Approximation coefficients - Level 3')
subplot(2,2,1);
imshow(H3)
title('Horizontal Cooefficients')
subplot(2,2,2);
imshow(V3)
title('Vertical Cooefficients')
subplot(2,2,3);
imshow(D3)
title('Diagonal Cooefficients')
subplot(2,2,4);
imshow(approx3,[])
title('Approximation Cooefficients')

%Level 4:
figure("Name",'3.4 - LEVEL 4');
sgtitle('Detail & Approximation coefficients - Level 4')
subplot(2,2,1);
imshow(H4)
title('Horizontal Cooefficients')
subplot(2,2,2);
imshow(V4)
title('Vertical Cooefficients')
subplot(2,2,3);
imshow(D4)
title('Diagonal Cooefficients')
subplot(2,2,4);
imshow(approx4,[])
title('Approximation Cooefficients')

%Level 5:
figure("Name",'3.4 - LEVEL 5');
sgtitle('Detail & Approximation coefficients - Level 5')
subplot(2,2,1);
imshow(H5)
title('Horizontal Cooefficients')
subplot(2,2,2);
imshow(V5)
title('Vertical Cooefficients')
subplot(2,2,3);
imshow(D5)
title('Diagonal Cooefficients')
subplot(2,2,4);
imshow(approx5,[])
title('Approximation Cooefficients')

%% Fuctions
%1.1.1
function F = dip_fft2(I)
M = size(I,1);
N = size(I,2);
chan = size(I,3);
m_u = (0:M-1);
n_v = (0:N-1);
w_m = exp(2*pi*1i.*((m_u.')*m_u)./M);
w_n = exp(2*pi*1i.*((n_v.')*n_v)./N);
F = zeros(M,N,chan);
for c = 1: chan
    F(:,:,c) = conj(w_m) * I(:,:,c) * conj(w_n);
end
end


function I = dip_ifft2(FFT)
M = size(FFT,1);
N = size(FFT,2);
chan = size(FFT,3);
m_u = (0:M-1);
n_v = (0:N-1);

w_m = exp(2*pi*1i.*((m_u.')*m_u)./M);
w_n = exp(2*pi*1i.*((n_v.')*n_v)./N);
I = zeros(M,N,chan);
for c = 1: chan
    I(:,:,c) = 1/(M*N)*(w_m * FFT(:,:,c) * w_n);
end
end


%1.1.2
function F_shift = dip_fftshift(FFT)
center_M = floor((size(FFT,1)-1)/2)+1;
center_N = floor((size(FFT,2)-1)/2)+1;
F_shift = [FFT(center_M+1:end,center_N+1:end,:),FFT(center_M+1:end,1:center_N,:);FFT(1:center_M,center_N+1:end,:),FFT(1:center_M,1:center_N,:)];
end


%1.2.1.e
function willy_without_bars =Free_Willy(Willy)
M = size(Willy,1);
N = size(Willy,2);
x=0:N-1;

fft_willy_prison = fftshift(fft2(Willy(1,:)));
single_side_fft = fft_willy_prison(N/2+2:end);
fVals=(1:N/2-1)/N;
[~,I] = max(abs(single_side_fft));
fx = abs(fVals(I))*N;

bars = 0.5 * sin(2*pi*fx.*x/N);
prison_bars=repmat(bars,M,1);
fft_prison_bars = fft2(prison_bars);
fft_willy = fft2(Willy);
fft_free_willy = fft_willy - fft_prison_bars;
willy_without_bars = real(ifft2(fft_free_willy));
figure;
imshow(willy_without_bars)
title('We freed Willy')
end

%1.2.2.e
function F_1D = sep_fft2(v1,v2)
F_1D = fftshift((fft(v1).')*fft(v2));
end

%---------------------------------------------------------------
function I_norm = normal(I)
I = double(I);
I_norm = (I - min(I(:)))/(max(I(:))-min(I(:)));
end

function pre = val2zero(beatle_DCT,range_min,gap)
beatle_DCT_range_a = beatle_DCT;
i = 1;
for a = range_min:gap:range_min+3*gap
    beatle_DCT_range_a(beatle_DCT_range_a > -a & beatle_DCT_range_a < a) = 0;
    beatle_IDCT_range_a = idct2(beatle_DCT_range_a);
    subplot(2,2,i)
    imshow(beatle_IDCT_range_a)
    title(['[-a,a] = [',num2str(-a), ',' ,num2str(a), ']'])
    i = i+1; 
end
numberOfZeros = sum(beatle_DCT_range_a(:)==0);
pre = numberOfZeros/(size(beatle_DCT_range_a,1)*size(beatle_DCT_range_a,2));
end

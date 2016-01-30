function plot_toy_example

%distilled_sgld = dlmread('toy-1d-distilled-sgld-50000.txt');
%sgld = dlmread('toy-1d-sgld-50000.txt');
pbp = dlmread('predict_PBP_test.txt');
distilled_sgld = dlmread('regression_DSGLD_other2.txt');
sgld = dlmread('regression_SGLD.txt');
groundtruth_train = dlmread('toy_data_train.txt');
groundtruth_test = dlmread('toy_data_test.txt');
groundtruth_test_whole = dlmread('toy_data_test_whole.txt')
x_train = groundtruth_train(:, 1);
y_train = groundtruth_train(:, 2);
x_test = groundtruth_test(:, 1);
y_test = groundtruth_test(:, 2);
m = sgld(:, 1);
s = sgld(:, 2);

single_plot(groundtruth_train(:, 1), groundtruth_train(:, 2), groundtruth_test_whole(:, 1), groundtruth_test_whole(:, 2), sgld(:, 1), sgld(:, 2))
single_plot(groundtruth_train(:, 1), groundtruth_train(:, 2), groundtruth_test_whole(:, 1), groundtruth_test_whole(:, 2), distilled_sgld(:, 1), distilled_sgld(:, 2))
single_plot(groundtruth_train(:, 1), groundtruth_train(:, 2), groundtruth_test_whole(:, 1), groundtruth_test_whole(:, 2), pbp(:, 1), pbp(:, 2))

end

function single_plot(x_train, y_train, x_test, y_test, m, s)
figure;
ylabel('Y')
xlabel('X')
[x_train, sortXtrain] = sort(x_train);
[x_test, sortI] = sort(x_test);
y_train = y_train(sortXtrain);
y_test = y_test(sortI);
m = m(sortI);
s = s(sortI);
axis([-6 6 -100 100]);
hold on
m_plus = m+3*sqrt(s);
m_minus = m-3*sqrt(s);
hold on
plot(x_test,m_plus,'-g', 'linewidth', 0.5)
plot(x_test,m_minus,'-g', 'linewidth', 0.5)
alpha(.01);

h = fill( [x_test' fliplr(x_test')],  [m_plus' fliplr(m_minus')], 'k', 'edgecolor','none');
set(h,'facealpha',.25)


%alpha(.25);

plot(x_train,y_train,'*r')
plot(x_test,m,'-b','linewidth',2)
plot(x_test,y_test, '-k', 'linewidth', 2)

end    
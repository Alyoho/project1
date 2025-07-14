import numpy as np
import matplotlib.pyplot as plt

def calculate(pred_array, true_array, i, seq_length=90, output_length=90):
    # 正确截取目标区间的真实值 [seq_length : seq_length+output_length]
    true_values = true_array[i, :]
    mae = np.mean(np.abs(pred_array[i,:] - true_values))
    mse = np.mean((pred_array[i,:] - true_values)**2)
    return mae,mse

def draw_one_sample(pred_array, true_array, model_name, i=0, seq_length=90, output_length=90):
    pred = pred_array[i]
    true = true_array[i]

    
    plt.figure(figsize=(8, 4))
    # 调整时间轴范围
    plt.plot(true, label='Target True Values', color='green', marker='o', markersize=3)
    plt.plot(pred, label='Predicted Values', color='red', linestyle='--', marker='x', markersize=3)
    plt.legend()
    plt.title(f'Prediction vs True Values for Sample {i+1}')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.grid()
    plt.savefig(f'figures/sample_{i+1}_{model_name}_prediction.png')  # 新建figures目录存放图片
    plt.close()

def postprocess(pred_array,true_array,model_name='transformer',seq_length=90, output_length=90):
    std_devs = np.std(pred_array, axis=0)
    mean_values = np.mean(pred_array, axis=0)
    
    with open('evaluation_results.txt', 'w', encoding='utf-8') as f:
        f.write('5轮训练平均值:\n') 
        num_samples = true_array.shape[0]
        val_i = list(range(10, num_samples-1, 50))[:10]
        for i in val_i:
            mae, mse = calculate(mean_values, true_array, i=i, seq_length=seq_length, output_length=output_length)
            f.write(f'测试集上第{i+1}个样本的MAE: {mae:.4f}\n')
            f.write(f'测试集上第{i+1}个样本的MSE: {mse:.4f}\n\n')
            # 添加绘图参数
            draw_one_sample(mean_values, true_array, i = i,model_name=model_name)
            
            plt.bar(height = std_devs[i,:], label='std_devs',x=range(len(std_devs[i,:])))
            plt.legend()
            plt.title(f'std for Sample {i+1}')
            plt.xlabel('Time Steps')
            plt.ylabel('Values')
            plt.savefig(f'figures/sample_{i+1}_{model_name}_std.png')  # 保存标准差图片
            plt.close() 
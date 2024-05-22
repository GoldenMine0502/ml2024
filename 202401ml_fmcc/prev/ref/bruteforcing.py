from prev.constant import *
from prev.train import dcunet10


for max_len_candidate in range(165000, 200000, 1):
    train_dataset.max_len = max_len_candidate

    noisy_x, clean_x = train_dataset[0]
    noisy_x = noisy_x.to(DEVICE).unsqueeze(0)
    clean_x = clean_x.to(DEVICE).unsqueeze(0)

    try:
        dcunet10(noisy_x)
        print(max_len_candidate, end=',')
        # break
    except Exception as e:
        if max_len_candidate % 1000 == 0:
            print('exception', e, max_len_candidate, noisy_x.shape)
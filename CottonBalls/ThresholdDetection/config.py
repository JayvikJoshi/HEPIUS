import os

save_down = True

input_path = "/Users/jayvik/Documents/GitHub/HEPIUS/CottonBalls/datasets/"
output_path = "/Users/jayvik/Documents/GitHub/HEPIUS/CottonBalls/output/"

og_img_height = 768
og_img_width = 1024
train_img_height = 192
train_img_width = 256
sliding_window_height = og_img_height // 4
sliding_window_width = og_img_width // 4

epochs = 30
batch_size = 16
train_pct = 0.7
val_pct = 0.15
test_pct = 0.15
from PIL import Image

RGPA_FILE = Image.open('Input_Image_Path') #RGBA file: # 4 is the channel
RGPA_FILE.load()  # required for png.split()


background = Image.new("RGB", RGPA_FILE.size, (255, 255, 255))
background.paste(RGPA_FILE, mask=RGPA_FILE.split()[3])  # 3 is the channel

background.save('Output_Image_Path', 'JPEG', quality=90)

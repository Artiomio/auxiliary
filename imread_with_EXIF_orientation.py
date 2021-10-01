from PIL import Image, ExifTags
import numpy as np

def imread_with_EXIF_orientation(filename):
    image = Image.open(filename)

    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif=dict(image._getexif().items())
    #   exif = image._getexif()

        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
            #print("    Поворачиваем на 180 градусов")
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
            #print("Поворачиваем на 270 градусов")        
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)
            #print("Поворачиваем на 90 градусов")        

    except:
        pass
        #print("Нет EXIF-информации или другая ошибка")

    img_arr = np.array(image)
    image.close()
    return img_arr
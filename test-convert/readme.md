# Как скомпилировать

## 0. Добавить англоязычний языковой пакет в visual studio (без него vcpkg не работает)
![](https://user-images.githubusercontent.com/24369177/151754180-12c823ef-8dd5-42a7-ae6e-d71fda0f158e.png)
## 1. Установить [vcpkg](https://github.com/microsoft/vcpkg) (пакетный менеджер для visual studio)
Установить можно в любой директории
```
git clone https://github.com/microsoft/vcpkg
```
```
.\vcpkg\bootstrap-vcpkg.bat
```
```
.\vcpkg\vcpkg integrate install
```
## 2. Установить [libheif](https://github.com/strukturag/libheif) из [vcpkg](https://github.com/microsoft/vcpkg)
```
.\vcpkg\vcpkg install libheif:x64-windows
```
Теперь libheif добавлен во все проекты visual studio 

Для импорта либы
```
#include <libheif/heif.h>
```
## 3. Добавить [stb_image_write.h](https://github.com/fitophoto/ann/blob/main/test-convert/stb_image_write.h) в проект
![](https://raw.githubusercontent.com/grimkel/justInCase/main/dir/add.PNG?token=GHSAT0AAAAAABNQUHMYLYGLFWPFMMPUX7BKYRR7XVA)

Файл взят из [stb](https://github.com/nothings/stb)

## 4. Скомпилировать в visual studio
![](https://github.com/grimkel/justInCase/blob/main/dir/comp.PNG)

Теперь все должно работать.

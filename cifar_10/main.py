from fastapi import FastAPI, HTTPException, UploadFile, File
from torchvision import transforms
# from pydantic import BaseModel
import streamlit as st
from PIL import Image
import torch.nn as nn
import uvicorn
import torch
import io


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class CifarClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, image):
        image = self.first(image)
        image = self.second(image)
        return image


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CifarClassification().to(device)
model.load_state_dict(torch.load('model_CifarClassification_CIFAR_10.pth', map_location=device, weights_only=True))
model.eval()

app = FastAPI(title="CIFAR-10 Classifier")


@app.post('/predict')
async def check_image(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail='Empty file')

        img = Image.open(io.BytesIO(data)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            predicted_class = output.argmax(dim=1).item()

        return {"class": classes[predicted_class]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)



# st.title('🎨 CIFAR-10 Классификатор')
# st.write('Загрузите изображение (самолёт, машина, птица и т.д.), и модель попробует определить класс')
#
# uploaded_file = st.file_uploader(
#     "Выберите изображение",
#     type=['png', 'jpg', 'jpeg']
# )
#
# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert('RGB')
#     st.image(image, caption='Загруженное изображение', use_container_width=True)
#
#     if st.button('🔍 Распознать', type='primary'):
#         try:
#             img_tensor = transform(image).unsqueeze(0).to(device)
#
#             with torch.no_grad():
#                 output = model(img_tensor)
#                 predicted_idx = output.argmax(dim=1).item()
#                 confidence = torch.softmax(output, dim=1)[0][predicted_idx].item()
#
#             st.success(f'**Модель думает, что это: {classes[predicted_idx]}**')
#             st.info(f'Уверенность: {confidence:.1%}')
#
#             probs = torch.softmax(output, dim=1)[0]
#             st.bar_chart(dict(zip(classes, probs.cpu().numpy())))
#
#         except Exception as e:
#             st.error(f'Ошибка при обработке: {str(e)}')
# else:
#     st.info('👆 Загрузите изображение выше')
#
# st.caption('Модель обучена на датасете CIFAR-10')
import gradio as gr 
from gradio.themes import Base
import os
import pandas as pd
import torch
from torchvision import transforms
from src.models.base.model import Model
import albumentations
from config import model_names, model_paths


class Predictor():
    def __init__(self, model_paths, model_names):
        self.models = []
        self.device = 'cuda'
        for path, name in zip(model_paths, model_names):
            model = Model(name, pretrained=True)
            model.load_state_dict(torch.load(path, map_location=self.device))
            self.models.append(model)

    def preprocessing(self, image):
        transforms_val = albumentations.Compose([albumentations.Resize(224, 224),
                                                albumentations.Normalize()])
        convert_tensor = transforms.ToTensor()

        image = transforms_val(image=image)['image']
        image = convert_tensor(image)
        image = image.unsqueeze(0)
        image = image.cuda()
        return image

    def get_snapshot(self):
        """
        functions helps us get image from camera
        return: captured_image 
        """
        pass

    def post_processing(self, prediction):
        """
        post processing for predicted results
        """
        pass

    def predict(self, image):
        output = torch.zeros([1, 2]).to(self.device)        
        for model_index in range(len(self.models)):
            self.models[model_index].to(self.device)
            self.models[model_index].eval()

        image = self.preprocessing(image=image)
        with torch.no_grad():
            for model in self.models:
                print(3*"#####################\n")
                output += model(image)
                print(output)
                print(3*"#####################\n")
            output /= len(self.models)            
            output = output.to('cpu')

        return output


class Seaform(Base):
    pass


def clear_field():
    return None, None

    
def clear_image_field():
    image_input.clear()
    return True


def save_patients_info(name, surname):
    if not name or  not surname:
        return f"error name ={name} and surname ={surname} ", None, None
    else:
        file_name = "/home/makarov/skin-canser-recognition/user_data.csv"
        info_dataframe = pd.DataFrame({"Name" : name,
                                   "Surname" : surname,},
                                   index = [0])
        
        if os.path.exists(file_name):
            current_dataframe = pd.read_csv(file_name)
            result_dataframe = pd.concat([current_dataframe,info_dataframe],
                                         ignore_index=True,
                                         sort=False)
            
            result_dataframe.to_csv(file_name,
                                    index=False)
            return "data has been writen", None, None
            
        else:
            info_dataframe.to_csv(file_name,
                                  index=False)
            return "data has been writen", None, None 


seaform = Seaform()
model_predictor = Predictor(model_names=model_names,
                            model_paths=model_paths)


with gr.Blocks(theme=seaform) as demo:
    gr.Markdown("HEADER FOR OUR DEMO")
    with gr.Tab("Patient Information"):
        name_input = gr.Textbox(label='Name')
        surname_input = gr.Textbox(label='Surname')
        
        with gr.Row():
            save_button = gr.Button("Save information")
            clear_button = gr.Button("Clear")
        
        with gr.Accordion("Debug", 
                          open=False):
            gr.Markdown("/////////")
            debug_box = gr.Textbox(label="Message from csv operation")
            # temp_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.1, interactive=True, label="Slide me")
            # temp_slider.change(lambda x:x, [temp_slider])

    with gr.Tab("Get image"):
        with gr.Row():
            image_input = gr.Image(sources=['webcam','upload','clipboard'], type='numpy')
            output_text = gr.Textbox(label="shape")
            
            image_input.change(model_predictor.predict,
                                inputs=image_input,
                                outputs=output_text)  
                      
        with gr.Row():    
            prediction_button = gr.Button("Calculate probability")
            clear_image_button = gr.Button("Clear")
        output_box = gr.Textbox(label="Result",
                                interactive=False)    

    
    prediction_button.click()
    save_button.click(save_patients_info,[name_input,surname_input],[debug_box, name_input, surname_input])
    clear_button.click(clear_field,[], [name_input, surname_input])
    #image_button.click(calculate_probability,[],output_box)
    clear_image_button.click(clear_image_field,[], [])


demo.launch(show_error=True)
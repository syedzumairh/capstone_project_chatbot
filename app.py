# app.py
from flask import Flask, render_template, request, jsonify
import openai
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#!pip install Pillow
import re
import openai
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import linear_kernel
from collections import Counter
from wordcloud import WordCloud

from PIL import Image
from IPython.display import display
from IPython.display import display, Image


app = Flask(__name__)

# Load the dataset
df = pd.read_csv('gadget.csv')

# Set your OpenAI API key
openai.api_key = "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"

# Initialize PorterStemmer
ps = PorterStemmer()

def generate_user_tags(input_features):
            tags = []
            gadget_name = input_features['Gadget']

            tags.append(str(input_features['Gadget']))
            tags.append(str(input_features['Brand name']))
            tags.append(str(input_features['Price']))
            tags.append(str(input_features['colour']))

            if gadget_name == 'laptop':

                tags.append(str(input_features['Gadget']))
                tags.append(str(input_features['Brand name']))
                tags.append(str(input_features['Price']))
                tags.append(str(input_features['colour']))
                tags.append(str(input_features['RAM']))
                tags.append(str(input_features['SSD']))
                tags.append(str(input_features['Operating system']))    
                tags.append(str(input_features['Hard disk']))
                tags.append(str(input_features['Processor']))
                tags.append(str(input_features['Graphics Processor']))
                tags.append(str(input_features['Battery Life']))



            elif gadget_name == 'mobile':
                tags.append(str(input_features['Gadget']))
                tags.append(str(input_features['Brand name']))
                tags.append(str(input_features['Price']))
                tags.append(str(input_features['colour']))
                tags.append(str(input_features['RAM']))
                tags.append(str(input_features['Touch Screen']))
                tags.append(str(input_features['Battery Life']))
                tags.append(str(input_features['Rear camera']))
                tags.append(str(input_features['Front camera']))
                tags.append(str(input_features['Internal storage']))


            elif gadget_name == 'television':
                tags.append(str(input_features['Gadget']))
                tags.append(str(input_features['Brand name']))
                tags.append(str(input_features['Price']))
                tags.append(str(input_features['colour']))
                tags.append(str(input_features['Display Size']))
                tags.append(str(input_features['Screen Type']))
                tags.append(str(input_features['Resolution Standard']))
                tags.append(str(input_features['Smart TV']))
                tags.append(str(input_features['Resolution (pixels)']))

            elif gadget_name == 'headphoneandspeaker':
                tags.append(str(input_features['Gadget']))
                tags.append(str(input_features['Brand name']))
                tags.append(str(input_features['Price']))
                tags.append(str(input_features['colour']))
                tags.append(str(input_features['Type']))
                tags.append(str(input_features['Wired/Wireless']))
                tags.append(str(input_features['Compatible Devices']))

            return tags    

# Function to stem text using PorterStemmer
def stem(text):
            ps = PorterStemmer()
            y = []
            for i in text.split():
                y.append(ps.stem(i))
            return " ".join(y)



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        prompt = data.get('prompt', '')


        if 'exit' in prompt or 'quit' in prompt:
            return jsonify({'response': 'Goodbye!'})

        # Check if the user input is "Can you recommend the device"
        if prompt.lower() == "can you recommend the device":
            
            # ... (same recommendation code as in your original code)
            # Return the recommended items as JSON
            while True:
                
                # input_gadget = input("Enter the gadget: ")
                # input_brand_name = input("Enter the brand: ")
                # input_price = str(input("Enter the price: "))
                # input_colour = input("Enter the colour: ")

                input_gadget = data.get('gadget')
                input_brand_name = data.get('brand_name','')
                input_price = data.get('price')
                input_colour = data.get('colour')



                input_features = {
                        'Gadget': input_gadget,
                        'Brand name': input_brand_name,
                        'Price': input_price,
                        'colour': input_colour,

                    }

                if input_gadget == 'laptop':
                        
                        input_ram=data.get("ram") 
                        #= input("Enter the RAM: ")
                        input_ssd=data.get('ssd') 
                        #= input("Enter the SSD: ")
                        input_os=data.get("os") 
                        # = input("Enter the operating system: ")
                        input_hard_disk=data.get("hard_disk") 
                        #= input("Enter the hard disk: ")
                        input_processor=data.get("processor")
                        #= input("Enter the processor: ")
                        input_graphics_processor =data.get("graphics_processor") 
                        #input("Enter the graphics processor: ")
                        input_battery_life=data.get("battery")
                        #= input("Enter the battery life: ")

                        input_features['RAM'] = input_ram
                        input_features['SSD'] = input_ssd
                        input_features['Operating system'] = input_os
                        input_features['Hard disk'] = input_hard_disk
                        input_features['Processor'] = input_processor
                        input_features['Graphics Processor'] = input_graphics_processor
                        input_features['Battery Life'] = input_battery_life

            
                elif input_gadget == 'mobile':
                       # input_ram = input("Enter the RAM ")
                        #input_touch_screen = input("Enter Yes or No for Touch Screen: ")
                        #input_battery_life = input("Enter the Battery Capacity: ")
                        #input_rear_camera = input("Enter Rear camera: ")
                        #input_front_camera = input("Enter Front Camera: ")
                        #input_internal_storage = input("Enter Internal Storage: ")

                        input_ram=data.get("ram") 
                        input_touch_screen=data.get("touch_screen") 
                        input_battery_life=data.get("battery_life") 
                        input_rear_camera=data.get("rear_camera") 
                        input_front_camera=data.get("front_camera")
                        input_internal_storage =data.get("internal_storage") 

                        

                        input_features['RAM'] = input_ram
                        input_features['Touch Screen'] = input_touch_screen
                        input_features['Battery Life'] = input_battery_life
                        input_features['Rear camera'] = input_rear_camera
                        input_features['Front camera'] = input_front_camera
                        input_features['Internal storage'] = input_internal_storage

                elif input_gadget == 'television':
                       
                        #input_display_size = input("Enter the display size: ")
                        #input_screen_type = input("Enter the screen type: ")
                        #input_resolution_standard = input("Enter the resolution standard: ")
                        #input_smart_tv = input("Enter Yes or No for smart TV: ")
                        #input_resolution = input("Enter the resolution (pixels): ")

                        
                        input_display_size = data.get("display_size")
                        input_screen_type = data.get("screen_type")
                        input_resolution_standard = data.get("resolution_standard")
                        input_smart_tv = data.get("smart_TV")
                        input_resolution = data.get("resolution")


                        input_features['Display Size'] = input_display_size
                        input_features['Screen Type'] = input_screen_type
                        input_features['Resolution Standard'] = input_resolution_standard
                        input_features['Smart TV'] = input_smart_tv
                        input_features['Resolution (pixels)'] = input_resolution

                elif input_gadget == 'headphoneandspeaker':
                        
                        #input_type = input("Enter the type: ")
                        #input_wired_wireless = input("Enter Wired or Wireless: ")
                        #input_compatible_devices = input("Enter the compatible devices: ")

                        input_type = data.get("type")
                        input_wired_wireless = data.get("Wired_Wireless")
                        input_compatible_devices = data.get("compatible_devices")


                        input_features['Type'] = input_type
                        input_features['Wired/Wireless'] = input_wired_wireless
                        input_features['Compatible Devices'] = input_compatible_devices




                user_tags = generate_user_tags(input_features)
                user_tags = stem(" ".join(user_tags))

                cv = CountVectorizer()

                    ## Combine the gadget and Brand name columns into a new column
                df['Gadget_Brand'] = df['Gadget'] + ' ' + df['Brand name']

                    ## Fit and transform the item Tags and Gadget_Brand Column
                tag_vectors = cv.fit_transform(df['Tags'])
                gadget_brand_vectors = cv.transform(df['Gadget_Brand'])

                    ## Transform the user tags and Gadget_Brand into vectors which are specified by the user
                user_tag_vector = cv.transform([user_tags])
                user_gadget_brand_vector = cv.transform([input_gadget + ' ' + input_brand_name]) 

                    ## Calculate the cosine similarity between user tags and Item_tags
                similarities_tag = cosine_similarity(user_tag_vector, tag_vectors)

                    ## Calculate the cosine similarity between user gadget_model and item gadget_model
                similarities_gadget_brand = cosine_similarity(user_gadget_brand_vector, gadget_brand_vectors)

                    ## Increase weightage for gadget and brand similarity
                similarities_tag *= 0.5 
                similarities_gadget_brand *= 1.5

                    ## Combine the similarities
                similarities_combined = similarities_tag + similarities_gadget_brand

                    ## Get the indices of top n similar items
                top_indices = similarities_combined.argsort()[0][::-1][:6]

                chatbot_response = "Recommended Items:\n"
                chatbot_response = "Recommended Items:\n"

                response = []

                for i, index in enumerate(top_indices, 1):
                        chatbot_response += f"{i}. {df.iloc[index]['Brand name']}\n"
                        chatbot_response += df.iloc[index]['Model number'] + "\n"
                        chatbot_response += str(df.iloc[index]['Price']) + "\n"
                        chatbot_response += df.iloc[index]['Tags'] + "\n"

                        # Get the image URL from the 'ImageURL' column
                        image_url = df.iloc[index]['picture url']

                            # Display the item details first
                        print("Chatbot:", chatbot_response)

                        response.append({'brand_name':f"{i}. {df.iloc[index]['Brand name']}\n",'model_number':df.iloc[index]['Model number'],'price': str(df.iloc[index]['Price']),'tags':df.iloc[index]['Tags'],'image':df.iloc[index]['picture url']})

                        # Display the image
                        display(Image(url=image_url))

                        # Clear the chatbot_response variable for the next item
                        chatbot_response = ""

                print("Chatbot:", response)


                return jsonify({'response': response})

        else:
            # Make a request to GPT-3 with the 'text-davinci-003' engine and max tokens set to 100
            completion = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=100,
            )
            response = completion.choices[0].text

            return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug = True)

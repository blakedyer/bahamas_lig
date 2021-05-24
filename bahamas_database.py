import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time
# from dataclasses import fieldSite
import pydeck as pdk
import os
import seaborn as sns
import sqlite3
import rasterio
import base64
from PIL import Image
import io
from io import BytesIO
import streamlit.components.v1 as components

sns.set_context('poster',font_scale=1.2)
# sns.set_style('ticks')

from matplotlib import pyplot as plt

def get_table_download_link(df,location):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f"""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <a href="data:file/csv;base64,{b64}" download="{location}.csv" style='text-decoration: inherit;'>
    <button style="background-color: DodgerBlue;border: none;color: white;padding: 12px 30px;cursor: pointer;font-size: 20px;  display: block; 
  margin-left: auto; font-size:100%;
  margin-right: auto;
  width: 40%;"><i class="fa fa-download"></i> Download {location}</button>
    </a>
    
    """
    return href

def gallery_item(path_to_file,w,h,timestamp):
    caption=timestamp
    
#     path_to_file=path_to_file
    aspect_ratio = w/h
    if w>h:
        ws=[1800,1024]
        hs=[int(1800/aspect_ratio),int(1024/aspect_ratio)]
    elif w<=h:
        ws=[1800,1024]
        hs=[int(1800/aspect_ratio),int(1024/aspect_ratio)]
        
    html=f"""
    
    <a href="http://www.blakedyer.com/Gallery/large/{path_to_file}" data-size="{ws[0]}x{hs[0]}" data-med="http://www.blakedyer.com/Gallery/http://www.blakedyer.com/Gallery/large/{path_to_file}" data-med-size="{ws[1]}x{hs[1]}" class="demo-gallery__img--main" data-author="Blake Dyer">
				<img src="http://www.blakedyer.com/Gallery/thumb/{path_to_file}" alt="" id="thumb" />
				<figure>{caption}</figure>
			</a>
            
        """
    return html


st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 60vw;
    }}
</style>
""",
        unsafe_allow_html=True,
    )
    

# st.set_page_config(page_title="database",
#                     layout="wide",
#                     initial_sidebar_state="expanded",)



df = pd.read_csv('data/field_data_checklist.csv',sep='\t',encoding='utf-7')

GPS=pd.read_csv('data/processed_GPS.csv').dropna()
    # GPS
GPS['lat']=pd.to_numeric(GPS['Latitude'])
GPS['lon']=pd.to_numeric(GPS['Longitude'])
GPS['Appx Height'] = np.round(GPS['Height (MSL)'],2)
GPS['Appx Height STD'] = np.round(GPS['Height (MSL) STD'],2)
    

section_list = df['Section']
section_list = [l for l in section_list if l[0]!='.']
# option = st.select_slider('Which section?',section_list)
st.sidebar.markdown('## Bahamas Sea Level Field Data Repository')
option = st.sidebar.select_slider('Slide to select a field location:',section_list,value='B1112')
st.sidebar.markdown('## '+str(option))
st.sidebar.markdown(str(df[df['Section']==option]['Comment'].values[0]))

subdf = df[df['Section']==option]
desc_str = subdf['Description'].values.astype(str)[0]
if desc_str == 'nan':
    st.sidebar.write('No locality description provided')
else:
    st.sidebar.write(desc_str)
    
conn = sqlite3.connect('lib_from_darktable.db')
c = conn.cursor()
c.execute('SELECT keywords FROM photos')
data=c.fetchall()
keywords=set(' '.join([elt[0] for elt in data]).split(' '))
conn.close()

with st.beta_expander("Location and GPS measurements:"):
    
    
    sub_GPS=GPS[GPS['Comment'].str.contains(option,na=False,case=False)]
    if len(sub_GPS)>0:
    
        st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v10",
                 initial_view_state=pdk.ViewState(
                     latitude=np.mean(sub_GPS['lat'].dropna()),
                     longitude=np.mean(sub_GPS['lon'].dropna()),
                     zoom=7.7,height=200
                 ),
                 layers=[pdk.Layer(
                         'ScatterplotLayer',
                         data=GPS.fillna(0),
                         get_position='[lon, lat]',
                         getFillColor='[155, 155, 155, 255]',
                         getLineColor='[0,0,0,255]',
                         radiusScale=1,
                         radiusMinPixels=5,
                         radiusMaxPixels=25,
                         stroked=True,
                         filled=True,
                         lineWidthMinPixels=1,
                         lineWidthMaxPixels=1,pickable=True),
                         pdk.Layer(
                         'ScatterplotLayer',
                         data=sub_GPS.fillna(0),
                         get_position='[lon, lat]',
                         getFillColor='[255, 255, 255, 255]',
                         getLineColor='[0,0,0,255]',
                         radiusScale=1,
                         radiusMinPixels=5,
                         radiusMaxPixels=25,
                         stroked=True,
                         filled=True,
                         lineWidthMinPixels=1,
                         lineWidthMaxPixels=1,pickable=True)
                        ], height=10,
                    tooltip={
               "html": " <b>Notes:</b> {Comment}<br/> <b>Height:</b> {Appx Height} ± {Appx Height STD} m<br/>",
               "style": {
                    "backgroundColor": "#2c3e50",
                    "color": "white"
               }, 
            }),use_container_width=False)


        c1, c2 = st.beta_columns((1, 1))


        c1.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/satellite-v9",
                 initial_view_state=pdk.ViewState(
                     latitude=np.mean(sub_GPS['lat'].dropna()),
                     longitude=np.mean(sub_GPS['lon'].dropna()),
                     zoom=17.7,
                 ),
                 layers=[pdk.Layer(
                         'ScatterplotLayer',
                         data=GPS.fillna(0),
                         get_position='[lon, lat]',
                         getFillColor='[155, 155, 155, 255]',
                         getLineColor='[0,0,0,255]',
                         radiusScale=1,
                         radiusMinPixels=5,
                         radiusMaxPixels=25,
                         stroked=True,
                         filled=True,
                         lineWidthMinPixels=1,
                         lineWidthMaxPixels=1,pickable=True),
                         pdk.Layer(
                         'ScatterplotLayer',
                         data=sub_GPS.fillna(0),
                         get_position='[lon, lat]',
                         getFillColor='[255, 255, 255, 255]',
                         getLineColor='[0,0,0,255]',
                         radiusScale=1,
                         radiusMinPixels=5,
                         radiusMaxPixels=25,
                         stroked=True,
                         filled=True,
                         lineWidthMinPixels=1,
                         lineWidthMaxPixels=1,pickable=True)
                        ], width='10%', height=100,
                    tooltip={
               "html": " <b>Notes:</b> {Comment}<br/> <b>Height:</b> {Appx Height} ± {Appx Height STD} m<br/>",
               "style": {
                    "backgroundColor": "#2c3e50",
                    "color": "white"
               }, 
            }
             ),use_container_width=True)

    #     with c2:
    #         'Written description of the field location'
        frame_to_show = sub_GPS[['Comment','Latitude','Longitude','Appx Height','Appx Height STD']]
        frame_to_show=frame_to_show.set_index('Comment')
        c2.write(frame_to_show)
        c2.markdown(get_table_download_link(sub_GPS,option), unsafe_allow_html=True)
    else:
        st.write('No GPS data for this location')

with st.beta_expander("Orthophoto:"):
#     st.write('only B1123 (jpg, not transparent) and B1112 (png, transparent) orthos uploaded so far')
    
    try:
        HtmlFile = open("ortho_html/"+option+".html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code,height=600)
    except FileNotFoundError:
        st.markdown('No orthophoto dataset available (yet). Only B1112 and B1123 available for now.')
    


#     st.write('not yet implemented -- will pull from photoscan reconstructions')
#     IMG_URL = '/limestone/Data/Sections/B1123/ortho.tif'
#     IMG_URL = '/limestone/Data/Sections/B1123/tiles/16/19098/37119.png'
    
    
#     dataset = rasterio.open(IMG_URL)
    
#     with open(IMG_URL, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
    
#     # Specifies the corners of the image bounding box
#     BOUNDS = [
#         [dataset.bounds.left, dataset.bounds.bottom],
#         [dataset.bounds.left, dataset.bounds.top],
#         [dataset.bounds.right, dataset.bounds.top],
#         [dataset.bounds.right, dataset.bounds.bottom],
#     ]
#     st.write(BOUNDS)
#     Image.MAX_IMAGE_PIXELS = 933120000

#     image = Image.open(IMG_URL)
    
# #     image.thumbnail((image.size[0],image.size[1]/)) 


#     buffered = BytesIO()
#     image.save(buffered, format="PNG")
#     img_str = base64.b64encode(buffered.getvalue())

#     img_base64 = bytes("data:image/png;base64,", encoding='utf-8') + img_str
    
#     # base64-encoded image string of a red dot
#     IMG_URL = '"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="'



#     bitmap_layer = pdk.Layer("BitmapLayer", data=None, image='"'+img_base64.decode('utf-8')+'"', bounds=BOUNDS, opacity=1)

#     view_state = pdk.ViewState(
#         latitude=(dataset.bounds.top+dataset.bounds.bottom)/2,
#         longitude=(dataset.bounds.left+dataset.bounds.right)/2,
#         zoom=15, max_zoom=25,
#     )
    
#     r = pdk.Deck(bitmap_layer, initial_view_state=view_state, map_provider="mapbox", map_style=pdk.map_styles.SATELLITE)
    
#     print(view_state.zoom)

#     st.pydeck_chart(r,use_container_width=True)




search_term = 'None'
with st.beta_expander("Field photos"):
    sub_term=st.text_input("Filter keywords:", option).lower()
    
    keywords=list(keywords)
    keywords.sort()
    keywords.insert(0,'None')
    
    filt = [key for key in keywords if sub_term in key.lower()]
    if option in filt:
        index=filt.index(option)
    else:
        index=0
    result = st.selectbox('Keyword Select:',filt,index)
    
    if st.button('Load photos'):
        search_term = result
        
        conn = sqlite3.connect('lib_from_darktable.db')
        c = conn.cursor()

        sqlite_select_query = "SELECT filename, w, h, timestamp FROM photos WHERE keywords LIKE ?"
        c.execute(sqlite_select_query,('%'+search_term+'%',))
        data=c.fetchall()
        COLUMN=0
        file_subset=[elt[COLUMN] for elt in data]
        COLUMN=1
        file_ws=[int.from_bytes(elt[COLUMN],byteorder='little') for elt in data]
        COLUMN=2
        file_hs=[int.from_bytes(elt[COLUMN],byteorder='little') for elt in data]
        COLUMN=3
        file_caption=[elt[COLUMN] for elt in data]

        conn.close()


        if (len(file_subset)>1):

            gallery = ''
            for f,w,h,c in zip(file_subset,file_ws,file_hs,file_caption):

                gallery+=gallery_item(f,w,h,c)


            mid=f"""

                    <div id="demo-test-gallery" class="demo-gallery">

                        {gallery}

                    </div>

            """

            HtmlFile = open("gallery_pre.html", 'r', encoding='utf-8')
            source_code_pre = HtmlFile.read() 

            HtmlFile = open("gallery_post.html", 'r', encoding='utf-8')
            source_code_post = HtmlFile.read() 


            components.html(source_code_pre+mid+source_code_post,scrolling=True,height=600
            )

#         image_iterator = paginator("Navigate images of field location", file_subset, items_per_page=6*5, on_sidebar=False)



#         cols = st.beta_columns((.1,1,.1,1,.1,1,.1,1,.1,1,.1,1,.1))

#         indices_on_page, images_on_page = map(list, zip(*image_iterator))

#         for i,j in enumerate(images_on_page):
#             i=(i%6)*2+1
#             with cols[i]:
#                 st.image(j,use_column_width=True,caption=j.split('/')[-1])
        elif (len(file_subset)>0):

            page_format_func = lambda i: "Image "+str(i)

            selected_photo = 0

            j=file_subset[selected_photo]
            st.image(j,use_column_width=True,caption=j.split('/')[-1])

        else:
            'No photos available for section '+str(option)

# pick_img = st.sidebar.radio("Which image?", 
#            [x for x in range(1, len(images_on_page))])
# st.image(images_on_page[pick_img],use_column_width=True)
# st.image(images_on_page, width=103)


# lookup='B1104'

# sqlite_select_query = "SELECT filename, keywords FROM photos WHERE keywords LIKE ?"
# c.execute(sqlite_select_query,('%'+lookup+'%',))
# data=c.fetchall()
# COLUMN=0
# file_subset=[elt[COLUMN] for elt in data]
# len(file_subset)
# conn.close()


# from time import time
# t1 = time()
# file_ = open("/limestone/Data/Photos/exported_library/2017/07/02/14_49_36.jpg", "rb")
# contents = file_.read()
# data_url = base64.b64encode(contents).decode("utf-8")
# file_.close()
# t2 = time()

# st.write(t2-t1)

# t1 = time()
# components.html(
#     f'<img src="data:image/jpg;base64,{data_url}" alt="cat gif" width=100>',
# )
# t2 = time()
# st.write(t2-t1)
# selected_images=['20190606-11-21-28.jpg','20190606-11-21-28.jpg']



# HtmlFile = open("http://www.blakedyer.com/models/test.html", 'r', encoding='utf-8')
# source_code = HtmlFile.read() 


# components.html(source_code,scrolling=True,height=600
# )
with st.beta_expander("3D Asset (testing)"):
    components.iframe("http://www.blakedyer.com/models/B1112.html",height=600)




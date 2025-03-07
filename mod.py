#!/usr/bin/env python
# coding: utf-8

# In[12]:


import random
import numpy as np
import pandas as pd


# In[14]:


import gradio as gr


# In[16]:


import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[49]:


# Sample dataset
data = {
    "questions": [
        "Hello", "Hi", "How are you?", "What is your name?", 
        "Who made you?", "What can you do?", "Tell me a joke","Swadharma","Bhagavad Gita ","Karma","Enlightenment",
        "Ego","Liberation","what is detachment?","The Purpose of Life "," Desires","Meditation","Lust","Anger","Greed","Self-Reflection","Practice Detachment"
    ],
    "responses": [
        "Hi there!", "Hello!", "I'm fine, thanks for asking!", 
        "I'm a sharv.", "I was created by Ashok.", 
        "I can chat with you!", "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "the importance of adhering to one's Swadharma, or one's own duty and righteousness. Performing one's duty without attachment to the results is considered essential.",
        "This great book of philosophy has guided the whole humanity through ages."," Karma is the idea that this life is the result of mental and physical actions","The state where not abel to identify diffrence between you and universe",
        "The cause of suffering, the delusional idea of the self as a separate entity",
        "the four paths to spiritual liberation (Moksha): the path of knowledge (Jnana Yoga), the path of devotion (Bhakti Yoga), the path of selfless action (Karma Yoga), and the path of renunciation (Sannyasa).",
        "It’s the ability to look at things without being affected by pain and pleasure",
        "We may run all over the place chasing after sense-objects and be consumed in the world of desires and ego, but the sooner we realize that the only true purpose to life is God, the better chance we have to be free of suffering and torment.",
        "Desires come and go, but you remain a dispassionate witness, simply watching and enjoying the show.",
        "Monkey Mind Cannot Meditate,I know we speak a great deal about watching, witnessing and silently comprehending the mind.  But, lets be clear that the ability to concentrate is a prerequisite for meditation.  Concentration is not meditation, meditation is much more than that.  But, without being able to apply your mind to the act of steady witnessing and being constantly adrift and absorbed in thoughts, you will not be able to meditate.  And without meditation, you will not uncover the Truth.  Without Truth, there is no end to suffering.",
        "Represents excessive desires, particularly those that drive us away from our moral and spiritual values.",
        "A powerful emotion that clouds judgment, leading to rash decisions and destructive outcomes.",
        "The insatiable longing for more, whether it be wealth, power, or possessions, which fosters dissatisfaction and conflict.",
        "Recognize when these emotions arise. Acknowledge their presence without judgment, as awareness is the first step to overcoming them.",
        "Develop a sense of contentment and let go of excessive attachment to material desires. Meditation and mindfulness can help in cultivating this detachment.",
        
        
        
        
        
    ]
}


# In[ ]:





# In[52]:


df = pd.DataFrame(data)


# In[ ]:





# In[55]:


df


# In[57]:


nltk.download('punkt')


# In[59]:


vectorizer = TfidfVectorizer()


# In[61]:


X = vectorizer.fit_transform(df["questions"])


# In[63]:


y = df["responses"]


# In[65]:


model = LogisticRegression()


# In[67]:


model.fit(X,y)


# In[69]:


# Function to generate responses
def chatbot_response(user_input):
    X_input = vectorizer.transform([user_input])
    if np.max(X_input.toarray()) == 0:  # If input is unknown
        return random.choice(["I don't understand.", "Can you rephrase?", "I'm not sure how to respond."])
    return model.predict(X_input)[0]


# In[71]:


# Gradio interface
iface = gr.Interface(fn=chatbot_response, inputs="text", outputs="text", title="Simple ML Chatbot")
iface.launch()


# In[73]:


from joblib import dump


# In[75]:


dump(model ,"shrv2.joblib")


# In[ ]:





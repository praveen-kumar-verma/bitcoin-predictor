#!/usr/bin/env python
# coding: utf-8

# In[5]:


#importing libraires
from bs4 import BeautifulSoup
import requests
import time
import smtplib
import ssl
from email.mime.text import MIMEText as MT
from email.mime.multipart import MIMEMultipart as MM


# In[11]:


#create a function to get the price of a cryptocurrency
def get_crypto_price(coin):
    #get URL
    url= "https://ycharts.com/indicators/"+coin+"_price"
    
    #making a request to the website
    HTML =requests.get(url)
    
    #Parse the HTML
    soup = BeautifulSoup(HTML.text, 'html.parser')
    
    #Find the current price
    text = soup.find("div", attrs={'class':'key-stat-title'}).text
    
    #return text
    return text


# In[12]:


get_crypto_price('bitcoin')


# In[ ]:


#storing the email adrres for the receiver and the sender and store the pass of sender
receiver = 'poornaasthana891@gmail.com'
sender = 'growwithguru2@gmail.com'
sender_password = 'Guruji@#190'


# In[ ]:


#creating a function to send email
def send_email(sender, receiver, sender_password, text_price):
    #createing a MIMEMultipart object
    msg = MM()
    msg['Subject'] = "New Crypto Price Alert! "
    msg['From'] = sender
    msg['To'] = receiver
    
    #Create the HTML for the message
    HTML = """
    <html>
       <body>
          <h1>New Crypto Price Alert!</h1>
          <h2>"""+text_price+"""
          </h2>
        </body>
    </html>
    """
    #creating a MIMEText object
    MTObj = MT(HTML, "html")
    #attach the MIMEtext Object
    msg.attach(MTObj)
    
    #creating ssl context object
    SSL_context = ssl.create_default_context()
    #creating the secure (SMTP) secure mail transfer layer connection
    server = smtplib.SMTP_SSL(host="smtp.gmail.com", port=465, context=SSL_context)
    #login to the email
    server.login(sender, sender_password)
    #sending mail
    server.sendmail(sender,receiver, msg.as_string())


# In[ ]:


send_email(sender, receiver, sender_password, 'test')


# In[ ]:


#createing a function to send a alert
def send_alert():
    last_price = -1
    #creating an infinte loop to continuously send/showing the price
    while True:
        #choose the cryptocurrency
        coin='bitcoin'
        #geting the price of the crypto
        price = get_crypto_price(coin)
        #check is the price has changed
        if price != last_price:
            print(coin.capitalize()+' price: ', price)
            price_text = coin.capitalize()+' is '+price
            send_email(sender, receiver, sender_password, price_text)
            last_price = price #update the last price
            time.sleep(3)
            


#sending alert
send_alert()







# write a langchain simple code 
from langchain import OpenAI
# Initialize the OpenAI model
model = OpenAI(model="gpt-3.5-turbo")
# Generate a response
response = model("What is the capital of France?")


# write a code to remove stop words from a sentence 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Sample sentence
sentence = "This is a sample sentence, showing off the stop words filtration."
# Tokenize the sentence
words = word_tokenize(sentence)
# Get the list of stop words
stop_words = set(stopwords.words('english'))
# Remove stop words from the sentence
filtered_sentence = [word for word in words if word.lower() not in stop_words]
print("Original sentence:", sentence)
print("Filtered sentence:", " ".join(filtered_sentence))

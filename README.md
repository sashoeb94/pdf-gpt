# PDF-GPT
Small program to feed PDF into Chat GPT model and run queries against it.


# Requirements
- Python 3 installed
- Run `pip install -r requirements.txt` to install the libraries
- Get your OpenAI API Key from the [site](https://platform.openai.com/account/api-keys) and add it to the constants.py file.

# Usage
```
python ask-gpt.py "Your query here" path/to/file.pdf startPage endPage
```
## Example - 
```
python ask-gpt.py "How do stock markets work? Summarize the answer in bullet points." marketguide.pdf 5 6
```

*Remember - Garbage in, Garbage out.*

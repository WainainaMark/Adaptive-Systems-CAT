import asyncio
from googletrans import Translator #use of google Translate API
from langdetect import detect #use of langdetect module to detect the language of the text to be translated


async def translate_text(text, src_lang, dest_lang):
    translator = Translator() #declaring a new Translator model
    translation = await translator.translate(text, src=src_lang, dest=dest_lang) #passing the arguments which include the text to be translated, the original language of the text, and the destination language of the translated text
    return translation.text # returns the translated text


async def main(text): #the function that runs the translation
    text = text
    lang = detect(text) #detects the type of language used in the text
    if lang == "sw":
        translated_to_english = await translate_text(text, "sw", "en")
        print(f"Swahili text: {text} \nTranslated English Text: {translated_to_english}") #Outputs the translated text
    elif lang == "en":
        translated_to_swahili = await translate_text(text, "en", "sw")
        print(f"English text: {text} \nTranslated Swahili Text: {translated_to_swahili}") #Outputs the translated text


# Run the async function
text = input("Insert the text to translate: ")
asyncio.run(main(text)) #runs the main function

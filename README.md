# PersianTelegramData
<b> Persian Telegram Data gathered from 8 July 2021 to 22 July 2021 </b>

This dataset contains six columns:
<br>
•	<b>context</b>: the text which is sent
<br>
•	<b>sender_username</b>: id of telegram channel
<br>
•	<b>sender_name</b>: name of telegram channel
<br>
•	<b>keywords</b>: list of keywords 
<br>
•	<b>hashtags</b>: hashtags used in the context
<br>
•	<b>send_time</b>: send time of the message in UTC DateTime
<br>

<b>How To Detect Keywords:</b>
<br>
We use <b>bert</b>( a contextualized word embedding based on Transformer) to convert words to meaningful vectors. The words that have the most cosine similarity to the context are keywords. To do this, we extract some candidate words and preprocess the context.
Preprocessing has these functions:
<br>
1.	Normalizing the context using Hazm library
2.	Tokenizing
3.	Using POS tagger to find Verbs in context 
4.	Detect stop words (words and their stemmed form must not be stop words)


Words that are not verbs, stopwords, and numbers can be a keyword



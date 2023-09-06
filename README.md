# Machine Translation for Biblical Texts
- [Machine Translation for Biblical Texts](#machine-translation-for-biblical-texts)
- [My Initial Thoughts](#my-initial-thoughts)
  - [Research Questions](#research-questions)
  - [Project Quality and Outcome Considerations](#project-quality-and-outcome-considerations)
  - [Additional Projects](#additional-projects)
- [Research Quetion: 1](#research-quetion-1)
- [Sample Translation](#sample-translation)
  - [Original Greek to English](#original-greek-to-english)
  - [Original Greek to Iraq Arabic, Mesopotamian \[acm\]](#original-greek-to-iraq-arabic-mesopotamian-acm)
- [Discussion](#discussion)

# My Initial Thoughts
While the field of machine translation research is still in a growth phase, it has matured enough to be used to build tools that could have an enormous benefit for the mission of bringing God's word to every language.

The primary challenge in any machine learning project lies in acquiring the necessary data. Companies like Google and Meta invest thousands of research hours each year attempting to address questions such as how to translate English internet content into languages that have never been translated before. They have a huge challenge in either sourcing data to train models on or in developing methodologies that do not require a lot of training data.  Additionally, they explore the possibility of creating devices capable of automatic translation while speaking (See [Additional Projects](#additional-projects) section).

Here, the goal of translating the Bible into other languages provides a unique advantage. There are numerous examples of well-established translations available. While there may not be enough, people have been working on this endeavor in some capacity for over 2000 years. We possess excellent examples that can be employed to train machine learning models. Furthermore, our output space is limited and compact. Instead of the daunting task of translating all of Wikipedia, we are dealing with only approximately 700k-800k words of the Bible. We understand our problem space well, and setting boundaries greatly enhances the effectiveness of machine learning algorithms.

Meta AI has developed a model capable of translating around 200 languages, including "150" low-resource languages. This model holds promise for translating scripture.

## Research Questions

1. How many languages can the Meta's out-of-the-box model translate that do not yet have a complete Bible translation?
2. Can the model be easily fine-tuned to improve translation accuracy for longer text sections?
   - The current goal of meta's research is to use one model for a lot of translations. We do not need that. We could have 7000k models because all we need is a model to do a good job translating from one language to another and then it is done.
   - The model was originally trained with input lengths not exceeding 512 tokens, potentially resulting in quality degradation when translating longer sequences. Fine tuning it from one-to-one language translations may increase the quality of longer input strings.
   - Given that scripture often relies on context that spans multiple sentences, longer input lengths may be necessary for higher translation quality.
   - Exploring translation windowing could potentially address this challenge.
3. Can fine-tuning a model with related languages, alongside a set of minimal translation samples, produce high-quality translations for new languages?
   - Is it feasible to provide individuals on the ground with a minimal set of passages or phrases to translate, serving as input for generating quality translations?
   - It is possible that a minimally viable set of passages for training a model in new languages could be identified through semantic rankings of scripture and existing "good" translations produced by humans.
4. While a model can take us a significant way in translation, can we relax quality restrictions to achieve approximately 80% accuracy  (for example) and then rely on human feedback from missionaries in the field to complete the translation?
   - Active Learning, which incorporates feedback from humans to iteratively improve output quality, can be employed in this context.
   - If an interim goal is to equip missionaries working on translations for new languages, processes like Active Learning could greatly expedite their work as it currently stands.

## Project Quality and Outcome Considerations

- What are the potential risks or unintended consequences of relying heavily on machine translation for scripture, and how can these risks be mitigate?
- How much involvement of linguists, theologians, and native speakers of target languages be integrated into the machine translation process?
- What are the potential challenges and nuances in translating scriptures that may not apply to other types of content? How can machine translation systems address these challenges effectively?
- Are there specific linguistic features, idioms, or cultural references present in bilical texts that might pose difficulties for machine translation models? How can these be addressed to ensure accurate translations?
- Can machine translation models take into account different theological interpretations and nuances in biblical texts, allowing for variations in translation based on different christian traditions? In other words, We do not want to narrow the understanding of "open-handed" theological ideas from a particular tradition into the text.

## Additional Projects

- How can machine translation technology be integrated with other tools or technologies, such as speech synthesis, to provide access to the scriptures for individuals who may have limited literacy skills in their native language? There are models designed for this purpose. I have tested some and they have suprised me on good some of the audio output sounds. While some may sound slightly robotic, others perform quite well. The question is, how many languages can these models effectively handle? The number is likely in the tens, if not lower. However, fine-tuning is a potential solution, and it might be achievable with a minimal set of examples. This question would need more research and advancement within the community.

# Research Quetion: 1

How many languages can the Meta's out-of-the-box model translate that do not yet have a complete Bible translation?

To answer this question I am going to use two datasets:

1. Progress.bible Data found originally at this [website](https://progress.bible/data/) compiled June 2023

- [Link to Google Sheet](https://docs.google.com/spreadsheets/d/1_xVfq1A7p3GnGTjCz9Q-wG0oj3TWI9jyzdZmuwGhhHM/edit#gid=0)

2. Meta's own list of supported languages, found [here](https://ai.meta.com/research/no-language-left-behind/) and [here](https://github.com/facebookresearch/flores/blob/main/flores200/README.md).

Full code can be found on [this](https://drive.google.com/file/d/1n4wE-5H4VYjT3kmBEleqVR8wIauvFKwA/view?usp=sharing) colab notebook. 


Load Progress.bibles data
```python
import pandas as pd
from google.colab import data_table
import typing
from typing import Tuple

language_list = pd.read_csv('ProgressBible_Language_List.csv', sep=',')
data_table.DataTable(language_list)
```

Load the list of supported languages from Meta's model
```python
supported_list = pd.read_csv('flores-200.tsv', sep='\t')
data_table.DataTable(supported_list)
```

Generate list of supported languages that do not have a full bible as well as the potential number of people this could effect. 
```python
multipliers = {
        'K': 1_000,
        'M': 1_000_000,
    }

def numeric_helper(text):
  # Remove any non-alphanumeric characters from the text
    cleaned_text = text.replace("+", "")
    # Extract the numeric part and suffix (if any)
    numeric_part = cleaned_text[:-1]
    suffix = cleaned_text[-1]
    # Multiply the numeric part by the appropriate multiplier based on the suffix
    if suffix in multipliers:
        return float(numeric_part) * multipliers[suffix]
    else:
        return float(cleaned_text)



def get_supported_languages(supported_list: pd.DataFrame, language_list: pd.DataFrame)-> Tuple[pd.DataFrame, float, float]:
    low_range = 0
    high_range = 0
    translate_now=[]
    # get the list of language that are supported
    lang_codes = ["[" + code + "]" for code in supported_list['Code'].tolist()]
    supported_lang = language_list['LanguageName'].str.contains('|'.join(lang_codes))
    for code in lang_codes:
      for index, row in language_list.iterrows():
        if (code in row['LanguageName']) & (row["Scripture"] != "Bible"):
          num_people = row['Population Group']
          if "-" in num_people:
            r=num_people.split()
            low_range += numeric_helper(r[0].strip())
            high_range += numeric_helper(r[2].strip())
          else:
            low_range += numeric_helper(num_people)
            high_range += numeric_helper(num_people)
          translate_now.append(row)
    final = pd.concat([pd.DataFrame([row]) for row in translate_now], ignore_index=True)
    return final, low_range, high_range


final_df, low_range, high_range = get_supported_languages(supported_list, language_list)

print("{num} Languages that could be helped right now!".format(num=len(final_df.index)))
print("{low} - {high} people could receive the word of God!".format(low=int(low_range), high=int(high_range)))
```

Output:
```
61 Languages that could be helped right now!
323,000,000 - 562,245,000 people could receive the word of God!
```


# Sample Translation

## Original Greek to English
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang="ell_Grek")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")

# Translate The first four verses from Romans 12 from greek to english
for article in ["παρακαλῶ οὖν ὑμᾶς ἀδελφοί διὰ τῶν οἰκτιρμῶν τοῦ θεοῦ παραστῆσαι τὰ σώματα ὑμῶν θυσίαν ζῶσαν ἁγίαν εὐάρεστον τῷ θεῷ τὴν λογικὴν λατρείαν ὑμῶν", "καὶ μὴ συσχηματίζεσθε τῷ αἰῶνι τούτῳ ἀλλὰ μεταμορφοῦσθε τῇ ἀνακαινώσει τοῦ νοός εἰς τὸ δοκιμάζειν ὑμᾶς τί τὸ θέλημα τοῦ θεοῦ τὸ ἀγαθὸν καὶ εὐάρεστον καὶ τέλειον", "λέγω γὰρ διὰ τῆς χάριτος τῆς δοθείσης μοι παντὶ τῷ ὄντι ἐν ὑμῖν μὴ ὑπερφρονεῖν παρ᾽ ὃ δεῖ φρονεῖν ἀλλὰ φρονεῖν εἰς τὸ σωφρονεῖν ἑκάστῳ ὡς ὁ θεὸς ἐμέρισεν μέτρον πίστεως", "καθάπερ γὰρ ἐν ἑνὶ σώματι πολλὰ μέλη ἔχομεν τὰ δὲ μέλη πάντα οὐ τὴν αὐτὴν ἔχει πρᾶξιν"]:
    inputs = tokenizer(article, return_tensors="pt")
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
    tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
```

The output:
> v1: Therefore, I urge you, brothers, in view of God's holiness, to offer your bodies as living sacrifices, holy and pleasing to God - this is your spiritual act of worship.
> 
> v2: Do not conform any longer to the pattern of this world, but be transformed by the renewing of your mind. Then you will be able to test and approve what God's will is - his good, pleasing and perfect will.
> 
> v3: For by the grace given me I say to every one of you: Do not think of yourself as wise, but think of yourself as wise, in accordance with the measure of faith God has given you.
>
> v4: Just as each of us has one body with many members, and these members do not all have the same function,

I would judge this as pretty good quality. It is pretty close to the NASB.

## Original Greek to Iraq Arabic, Mesopotamian [acm]

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang="ell_Grek")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")

# Translate The first four verses from Romans 12 from greek to english
for article in ["παρακαλῶ οὖν ὑμᾶς ἀδελφοί διὰ τῶν οἰκτιρμῶν τοῦ θεοῦ παραστῆσαι τὰ σώματα ὑμῶν θυσίαν ζῶσαν ἁγίαν εὐάρεστον τῷ θεῷ τὴν λογικὴν λατρείαν ὑμῶν", "καὶ μὴ συσχηματίζεσθε τῷ αἰῶνι τούτῳ ἀλλὰ μεταμορφοῦσθε τῇ ἀνακαινώσει τοῦ νοός εἰς τὸ δοκιμάζειν ὑμᾶς τί τὸ θέλημα τοῦ θεοῦ τὸ ἀγαθὸν καὶ εὐάρεστον καὶ τέλειον", "λέγω γὰρ διὰ τῆς χάριτος τῆς δοθείσης μοι παντὶ τῷ ὄντι ἐν ὑμῖν μὴ ὑπερφρονεῖν παρ᾽ ὃ δεῖ φρονεῖν ἀλλὰ φρονεῖν εἰς τὸ σωφρονεῖν ἑκάστῳ ὡς ὁ θεὸς ἐμέρισεν μέτρον πίστεως", "καθάπερ γὰρ ἐν ἑνὶ σώματι πολλὰ μέλη ἔχομεν τὰ δὲ μέλη πάντα οὐ τὴν αὐτὴν ἔχει πρᾶξιν"]:
    inputs = tokenizer(article, return_tensors="pt")
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["acm_Arab"])
    tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
```
The output:

> v1: 'إِذَنْ يَا اِخْوَتِي ، بِمَحْسَبِ مَا يُعْطِيهِ اللهُ لِيَسْعَى إِلَيْهِ ، أَرْجُوكُمْ أَنْ تُقَدِّمُوا أَنْفُسَكُمْ لَهُ ضَحِيَّةً حَيَّةً مُكَرَّسَةً وَيَرْضَى بِهَا اللهُ ، فَيَكُونُ هَذَا مِنْكُمْ عِبَادَةً رُوحِيَّةً.'

> v2: 'لا تَعِيشُوا حَسَبَ عَادَةِ هَذِهِ الدُّنْيَا ، بَلْ غَيِّرُوا أَنْفُسَكُمْ بِأَنْ تُجَدِّدُوا عُقُولَكُمْ. بِذَلِكَ تَعْرِفُونَ إِرَادَةَ اللهِ ، لأَنَّهَا صَالِحَةٌ وَمَقْبُولَةٌ وَكَامِلَةٌ.'

> v3: 'أَنَا أَقُولُ لَكُمْ بِوَاسِطَةِ نِعْمَةِ اللهِ الَّتِي أَعْطَاهَا لِي: لا تَعْتَبِرُوا أَنْفُسَكُمْ بَطَوِيِّينَ أَوْ مُتَعَقِّلا. بَلْ كُونُوا عُقَلاءَ بِمَا يَعْنِي الْعَقْلَ. فَكُلُّ وَاحِدٍ يَجِبُ أَنْ يَعْتَبِرَ نَصِيبَهُ مِنَ الإِيمَانِ الَّذِي أَعْ'

> v4: 'وَكَمَا أَنَّ فِي جِسْمٍ وَاحِدٍ أَعْضَاءً كَثِيرَةً ، وَلَكِنَّ كُلَّ الأَعْضَاءِ لَيْسَ لَهَا مَقَامٌ وَاحِدٌ ،'

I cannot assess the quality of this translation.


# Discussion
It's quite fascinating to witness text being translated. I'm not naive enough to believe that I'm the first person to have thought of the concept of using ML algorithms to translate scripture. I'm certain that others have been employing similar methodologies. I presume there exists quality issue for many languages. It's essential to always have a human reviewer involved; after all, we are dealing with the word of God, and accuracy is paramount. However, what I hope to demonstrate is that modern machine learning algorithms can take us a considerable distance. Perhaps the translation accuracy reaches around 80%, necessitating human review. I'm confident that the process of reviewing and making adjustments is far more manageable than starting from scratch.

There is immense potential to develop tools that can significantly aid the translation efforts for languages lacking any scriptural content.

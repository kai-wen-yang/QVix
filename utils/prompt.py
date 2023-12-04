prompt_hand_v1 = '''I need to answer the following main question about an image: {}

Your goal is to design 4 pre-questions. Pre-questions should focus on important contextual information in the image useful for answering the main question.

Here are the rules you should follow when listing the sub-questions:
Each pre-question should be short and easy to understand.
Pre-questions should focus on context visual clues of the image.
Pre-questions should provide clues to answer the main question.

Format Example:
Pre-question 1: xxxx
Pre-question 2: xxxx 
Pre-question 3: xxxx
Pre-question 4: xxxx'''


prompt_gptrewrite_v1 = '''I require assistance in formulating a response to a central inquiry regarding a specific image: 
{}

The task is to create 4 preliminary questions. These questions should zero in on crucial contextual details within the image that are pertinent to addressing the main inquiry.

Guidelines for the preliminary questions:

Each question must be concise and easily comprehensible.
They should concentrate on contextual visual elements present in the image.
These questions ought to offer insights that aid in responding to the main question.
Proposed Format:
Preliminary Question 1: xxxx
Preliminary Question 2: xxxx
Preliminary Question 3: xxxx
Preliminary Question 4: xxx'''

prompt_gptrewrite_v2 ='''I am tasked with addressing a primary inquiry regarding a specific image: 
{}

My objective is to formulate 4 preliminary questions. These questions are aimed at eliciting critical contextual details from the image, which are pivotal for comprehensively responding to the main inquiry.

Guidelines for crafting the preliminary questions:

Each question must be concise and easily comprehensible.
The focus should be on discerning visual cues within the image that offer context.
These questions are intended to unearth insights that facilitate answering the main question.
Formatted Example:
Preliminary Question 1: [Your Question Here]
Preliminary Question 2: [Your Question Here]
Preliminary Question 3: [Your Question Here]
Preliminary Question 4: [Your Question Here]'''


prompt_gptrewrite_v3 ='''I am required to address a primary inquiry related to a specific image: 
{}

The task is to formulate 4 preliminary questions. These questions are intended to extract key contextual details from the image that are crucial for responding accurately to the primary inquiry.

Guidelines for creating the preliminary questions:

Each question should be concise and straightforward for ease of understanding.
The focus should be on discernible contextual elements within the image.
These questions should aid in gathering insights necessary to address the primary inquiry.
Example of the format:
Preliminary Question 1: [Question text]
Preliminary Question 2: [Question text]
Preliminary Question 3: [Question text]
Preliminary Question 4: [Question text]'''


prompt_template = {
'prompt_hand_v1': prompt_hand_v1,
'prompt_gptrewrite_v1': prompt_gptrewrite_v1,
'prompt_gptrewrite_v2': prompt_gptrewrite_v2,
'prompt_gptrewrite_v3': prompt_gptrewrite_v3
}

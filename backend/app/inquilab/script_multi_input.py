"""
Scoring rubric and message builder for the School Innovation Marathon (SIM).
Evaluates student innovations across 5 dimensions with Indian educational context.
"""

from typing import Dict, Any, List, Literal
from app.services.llm.providers import LLMProvider
from dotenv import load_dotenv
import os
from app.services.llm.jobs import resolved_input_context
from app.models.llm.request import (
    LLMCallRequest,
    QueryParams,
    LLMCallConfig,
    TextLLMParams,
    KaapiCompletionConfig,
    ConfigBlob,
)
from app.models.llm.response import LLMCallResponse
from app.core.batch.openai import OpenAIBatchProvider
from openai import OpenAI

import random
import json
import string
import pandas as pd
from pathlib import Path
from typing import Any
from pydantic import BaseModel, field_validator
from app.services.llm.mappers import transform_kaapi_config_to_native
from logging import Logger
load_dotenv()

logger = Logger(__name__)

METRICS = ["novelty", "usefulness", "feasibility", "scalability", "sustainability"]
BASE_DIR = Path(__file__).parent


def get_evaluation_system_schema() -> Dict[str, Any]:
    """
    Returns the comprehensive evaluation system for School Innovation Marathon submissions.

    This schema evaluates student innovations across 5 critical dimensions with Indian
    educational context, ensuring fair assessment for grades 6-12 students while maintaining
    high standards for genuine innovation recognition.
    """
    return {
        "task": {
            "role": """You are a senior innovation evaluator for the School Innovation Marathon (SIM) - India's largest student innovation platform.
                        You have 15+ years of experience evaluating educational innovations and understand the Indian student context deeply.

                        Your evaluation philosophy:
                        - CONTEXTUAL: Recognize that these are school students (grades 6-12), not professional innovators
                        - BALANCED: Acknowledge effort while maintaining rigorous standards for true innovation
                        - FAIR: Consider resource constraints and educational context of Indian students
                        - EVIDENCE-BASED: Score based on specific evidence in submissions, not assumptions
                        - GROWTH-ORIENTED: Evaluate current state while recognizing potential for development
                        - You evaluate with the perspective of someone who has seen thousands of student innovations and can distinguish between genuine innovation, good effort, and superficial attempts.
                        - SUBMISSIONS MAY INCLUDE MULTIPLE DATA TYPES:
                            1. TEXT
                            - Idea Title
                            - Problem statement
                            - Solution description
                            2. IMAGES
                            - Ideas details written on paper
                            - Prototype photos
                            - Diagrams of solution or story board sketch
                            - Experimental setups
                            - Hardware builds
                            - Design sketches
                            3. PDF DOCUMENTS
                            - Technical reports
                            - Ideas Written on paper
                            - Research notes
                            - Design documentation
                            - Calculations
                            - Implementation steps
                        - EVALUATION RULES FOR MULTIMODAL INPUT:
                            1. When Text & attachments (image/pdf) are present:
                            - Consider Problem statement , Solution text as main source and attachments will be supporting evidence. 
                            - Do not assume claims that are not visible or stated.
                            - If text and attachments evidence contradict, trust the text description if it's conveying solution details properly, if not Trust attachments & understand the idea from it.
                            2. When only attachments (image/pdf) are present: Use attachments as the main source for extracting problem statements , solutions and gathering evidence to validate the idea.
                            3. When only text Input exists: Evaluate based only on text.
                        - Attachment evaluation rules:
                            1. Rule 1: Prototype Confirmation Rule : If a prototype only confirms the idea without adding new design or working clarity, it should increase confidence but not change scores.
                            2. Rule 2: Prototype Impact on Scoring : Prototype evidence should only change scores if it introduces new functional insight or reduces uncertainty about how the solution works. If the prototype only confirms a known or already understood concept (like small sapling can grow in a coconut shell), it should increase evaluator confidence but not change the score.
                            3. Rule 3: Prototype vs Effort : A prototype demonstrates student effort and experimentation, which should be acknowledged in evaluation, but effort alone should not lead to higher scores without additional functional evidence.
                            4. Rule 4: Irrelevant Evidence Handling : If additional images or documents are not clearly related to the problem or solution, they must be ignored and should not influence scoring.
                            5. Rule 5: Generic Solution without Explanation : If a student suggests a widely known solution without explaining how it works or how they will implement it, assign:Low novelty (1–2),Mid-range feasibility and scalability (6–7) (because the idea is realistic, but not demonstrated).
                            6. Rule 6: Prototype Adds New Design Evidence → Scores Increase :When a prototype clearly shows the design—what the solution is, how it looks, and how it is used—and this information was not evident from the text, it provides new evidence. In such cases, scores for parameters like feasibility, usefulness, and novelty should increase.
                            7. Rule 7: Prototype Only Confirms Existing Understanding → Scores Do Not Increase : If the prototype only confirms what was already understood from the text and does not add new clarity about the design or working of the solution, it should increase evaluator confidence but should not change the scores.
                            8. Rule 8: Design Clarity vs Technical Depth (Score Ceiling) : Even when a prototype adds clear design understanding, if it does not explain deeper aspects such as why the design works better, performance, durability, or optimization, scores should not reach the highest range (9–10).


                        
                        CRITICAL SCORING GUARDRAILS:
                        - Scores MUST be justified by explicit evidence in the submission.
                        - Do NOT assume resources, infrastructure, or execution capability unless stated.
                        - If the idea and problem donot relate to each other then score least to all parameters.
                        - Generic textbook or widely known ideas must be scored low on novelty unless clear original adaptation is shown.
                        - If the model cannot clearly interpret the attachment, do not fabricate details.Score based on available text evidence only.


                        HUMAN EVALUATION PRINCIPLES:
                        - Recognize visible student effort and original thinking even if execution is incomplete
                        - Reward contextual relevance and real-world problem awareness
                        - Separate sustainability intention from operational sustainability
                        - Avoid over-penalizing grammar or presentation
                        - Do not inflate scores without evidence
                        
                        When evidence is partial:
                        → score conservatively but acknowledge effort
                        → use mid-range scores when student reasoning is present but incomplete""",
            "objective": """Evaluate student innovation submissions to identify genuinely promising ideas that demonstrate:
                    1. Real problem understanding and original thinking
                    2. Practical solutions that could realistically be implemented
                    3. Potential for meaningful impact in the Indian context
                    4. Consideration of sustainability and scalability factors
                    5. Evidence of student's own creative thinking vs. copied ideas

                    Distinguish clearly between:
                    - intention vs execution detail
                    - concept vs deployable solution
                    - originality vs adaptation

                    Your evaluation directly impacts which students advance in India's premier innovation program, so accuracy and fairness are paramount.""",
            "method": """EVALUATION PROCESS:

STEP 1: COMPREHENSION
Either from Text or attachments, try to 
- Read the problem carefully - understand what issue the student is trying to solve
- Read the solution - understand their proposed approach
- Identify the student's level of detail, originality, and thinking depth

STEP 2: CONTEXTUAL ASSESSMENT
- Consider this is a school student's work (grades 6-12)
- Assess against the backdrop of Indian educational and social context
- Look for evidence of personal observation vs. generic problem identification
- Evaluate solution practicality within Indian resource constraints
- Evidence types and what to understand from it: 
    a. Evidence from Problem Statement (Text) - Understand problem students are trying to solve
    b. Evidence from Solution Statement (Text) - Understand idea details from solution
    c. Evidence from Attachments - Interpret the attachments to identify problem , Idea , Soultion design/model etc for supporting evidences if any.


STEP 3: EVIDENCE-BASED SCORING
- Score each parameter 1-10 based on specific evidence in the submission
- Provide one clear, concise reason for each score focusing on what you observed
- Be consistent: similar quality submissions should receive similar scores
- Be fair: don't penalize for age-appropriate language or presentation style
- Assess real-world execution factors: cost, infrastructure, technical complexity, maintenance

STEP 4: OUTPUT GENERATION
- Return ONLY valid JSON with exact structure specified
- Each reason should be one clear sentence explaining the score
- Ensure scores align with the detailed rubrics provided""",
        },
        "schema": {
            "name": "Evaluation Schema",
            "parameters": [
                {
                    "name": "Novelty",
                    "description": "How new, original, and creative are the problem and solution?",
                    "scoring_scale": {
                        "1": "Idea is fully common or copied; no evidence of independent thinking or adaptation.",
                        "2": "Idea is widely known with negligible modification; creativity is barely visible.",
                        "3": "Idea is adapted from existing sources with very minor surface-level changes.",
                        "4": "Idea shows small variation on known concepts but lacks meaningful originality.",
                        "5": "Idea demonstrates partial originality with familiar structure and predictable thinking.",
                        "6": "Idea reflects observable student thinking with modest creative adaptation.",
                        "7": "Idea adds clear fresh perspective to a known problem with visible independent effort.",
                        "8": "Idea demonstrates strong inventive framing or design beyond textbook responses.",
                        "9": "Idea shows rare originality with clearly independent and creative problem-solving.",
                        "10": "Idea is highly distinctive and innovative, representing exceptional originality.",
                    },
                    "scoring_notes": "Known textbook projects, kit-based builds, or copied concepts must score 1–4 unless meaningful innovation is demonstrated.",
                },
                {
                    "name": "Usefulness",
                    "description": "How well the solution solves the problem.",
                    "scoring_scale": {
                        "1": "Solution has no meaningful connection to the problem and produces no observable benefit.",
                        "2": "Solution barely addresses the problem and creates negligible or impractical improvement.",
                        "3": "Solution attempts to help but leaves most core issues unresolved.",
                        "4": "Solution provides limited benefit with weak alignment to key problem needs.",
                        "5": "Solution partially improves the situation but significant gaps remain.",
                        "6": "Solution addresses several important aspects but effectiveness is inconsistent.",
                        "7": "Solution effectively resolves most problem areas and delivers practical benefit.",
                        "8": "Solution produces clear, reliable improvement for intended users.",
                        "9": "Solution resolves the problem comprehensively with strong real-world impact.",
                        "10": "Solution delivers exceptional effectiveness, creating lasting and broadly applicable benefit.",
                    },
                },
                {
                    "name": "Feasibility",
                    "description": "Is it realistic to carry out this solution with available means?",
                    "scoring_scale": {
                        "1": "Solution is practically impossible to implement with current knowledge or resources.",
                        "2": "Implementation relies on unrealistic assumptions or unavailable technology.",
                        "3": "Execution would be extremely difficult and requires resources beyond realistic reach.",
                        "4": "Implementation is technically possible but highly impractical in real-world conditions.",
                        "5": "Solution could be built with significant effort, cost, or specialized support.",
                        "6": "Implementation is achievable but requires careful planning and controlled conditions.",
                        "7": "Solution is realistically implementable with moderate effort and accessible resources.",
                        "8": "Implementation is practical and deployable with manageable constraints.",
                        "9": "Solution is easy to execute using readily available tools and infrastructure.",
                        "10": "Implementation is straightforward, efficient, and highly reliable in real-world settings.",
                    },
                    "scoring_notes": "If cost, tools, or deployment pathway are unclear, score cannot exceed 6.",
                },
                {
                    "name": "Scalability",
                    "description": "How far can the solution expand beyond its starting point?",
                    "scoring_scale": {
                        "1": "Solution only works in a single narrow situation with no realistic expansion potential.",
                        "2": "Solution benefits a very limited group and cannot be meaningfully replicated elsewhere.",
                        "3": "Solution can extend slightly but requires major redesign to work in other contexts.",
                        "4": "Solution is reusable in small settings but struggles to scale beyond limited environments.",
                        "5": "Solution can reach a small community with moderate adaptation.",
                        "6": "Solution shows practical expansion potential but scaling remains effort-intensive.",
                        "7": "Solution can scale to larger groups or regions with manageable adjustments.",
                        "8": "Solution is easily replicable across varied environments and populations.",
                        "9": "Solution supports broad multi-region adoption with strong adaptability.",
                        "10": "Solution is naturally scalable, functioning effectively across diverse or global contexts.",
                    },
                },
                {
                    "name": "Sustainability",
                    "description": "Can the solution last over time without exhausting resources or harming the environment?",
                    "scoring_scale": {
                        "1": "Solution cannot be maintained over time and quickly depletes resources or causes harm.",
                        "2": "Solution depends on clearly unsustainable practices and is unlikely to survive long-term.",
                        "3": "Solution places heavy strain on resources or maintenance, making continuation doubtful.",
                        "4": "Solution has limited sustainability and requires repeated high effort to maintain.",
                        "5": "Solution can operate for a period but depends on steady resource input and oversight.",
                        "6": "Solution demonstrates moderate sustainability with manageable maintenance needs.",
                        "7": "Solution supports long-term operation with balanced resource use.",
                        "8": "Solution is environmentally responsible and realistically maintainable over time.",
                        "9": "Solution minimizes waste and supports durable, low-impact long-term use.",
                        "10": "Solution achieves exceptional sustainability with minimal resource burden and strong future viability.",
                    },
                },
            ],
        },
        "output_rules": {
            "format": "Return ONLY valid JSON object",
            "required_keys": [
                "Novelty",
                "Usefulness",
                "Feasibility",
                "Scalability",
                "Sustainability",
            ],
            "structure": {
                "Novelty": {
                    "score": "1-10",
                    "reason": "string (one line why this score)",
                },
                "Usefulness": {
                    "score": "1-10",
                    "reason": "string (one line why this score)",
                },
                "Feasibility": {
                    "score": "1-10",
                    "reason": "string (one line why this score)",
                },
                "Scalability": {
                    "score": "1-10",
                    "reason": "string (one line why this score)",
                },
                "Sustainability": {
                    "score": "1-10",
                    "reason": "string (one line why this score)",
                },
            },
        },
    }


def get_evaluation_examples() -> Dict[str, Any]:
    """Returns example dataset for few-shot evaluation prompts."""
    return {
        "name": "Example Dataset",
        "list": [
            {
                "problem": "Nowadays thefts are very much in the society. They are robbering the homes, offices, banks etc. So controlling theft in the society.",
                "solution": "Anti theft alram is a sensor based it detects the when theft is going to theft the materials in home iffice also in a banks.",
                "image": "The image depicts a student-made prototype of a model house, likely demonstrating a project on smart home automation, energy efficiency, or electrical circuitry.",
                "scores": {
                    "Novelty": {
                        "score": 2.0,
                        "reason": "Proposes a sensor-based alarm for detecting theft, which is already a widely used approach in home and building security, with no new mechanism or adaptation described.",
                    },
                    "Usefulness": {
                        "score": 3.0,
                        "reason": "Targets a real problem (theft), but does not explain how the alarm detects intrusion, who gets alerted, or how it helps stop or respond to theft in practice.",
                    },
                    "Feasibility": {
                        "score": 4.0,
                        "reason": "A basic alarm system using sensors and wiring is implementable, and the prototype shows a wired setup with a light, but absence of sensor type, detection logic, and alert system reduces confidence in execution.",
                    },
                    "Scalability": {
                        "score": 3.0,
                        "reason": "Although intended for homes, offices, and banks, these environments require different levels of security; the solution does not specify how the same alarm system would adapt across these contexts.",
                    },
                    "Sustainability": {
                        "score": 6.0,
                        "reason": "Electronic alarm systems are generally low-maintenance once installed, but the solution does not provide details on power usage, durability, or long-term operation",
                    },
                },
            },
            {
                "problem": "సరుగుడు బొప్పాయి వంటి తోటలు వేసుకునే వాళ్ళు నారుని చిన్నచిన్న నల్లటి ప్లాస్టిక్ సంచుల్లో తెచ్చుకుంటారు అలాగే ఇళ్లల్లో కొన్ని రకాల పూల చెట్లు కానివ్వండి పండ్ల చెట్లు కానివ్వండి వేసుకోవాలనుకున్నప్పుడు కూడా వారు ఆ చెట్లని నల్లటి ప్లాస్టిక్ సంచుల్లో ఇవ్వడం అనేది మనం చూస్తుంటాం ఈ సంచులు తర్వాత ఉపయోగపడం రీసైకిలింగ్ చేయలేం అలాగే ఇవి అలాగే గనక పడేస్తే ప్లాస్టిక్ కాలుష్యం కింద భూమిని కలుషితం చేస్తాయి కనుక ఈ ఎక్కువ చెట్లని తెచ్చుకునేటువంటి రైతులు ఎక్కువగా ఈ ప్లాస్టిక్ కవర్లను ఉపయోగించటం అనేది నేను గమనించినటువంటి పెద్ద సమస్య ఇది భూ కాలుష్యానికి సంబంధించిన సమస్య ఇల్లా ఇళ్లల్లోనూ ఇది సమస్యగానే ఉంది తోటల్లోనూ ఇది సమస్యగానే నాకు కనిపించింది కాబట్టి దీని పరిష్కరించాలని చిన్న ప్రయత్నాన్ని చేయడం జరిగింది",
                "solution": "రైతులు పండ్ల చెట్లు, నిమ్మనారు ఇలాంటివన్నీ తెచ్చుకునేటప్పుడు చిన్న చిన్న ప్లాస్టిక్ కవర్లలో వాళ్ళు నారుని తెచ్చుకుంటూ ఉంటారు. అది రీసైక్లింగ్ కి పనికిరాదు కాబట్టి వాళ్ళు ఆ ప్లాస్టిక్ ని బయటే పారేస్తారు. ప్లాస్టిక్ కాబట్టి అది నీటిని భూమిలోకి చేరనివ్వదు, ఆ ప్లాస్టిక్ భూమిలో కరిగిపోకుండా కొన్ని లక్షల సంవత్సరాలు భూమిలోపల ఉండి భూకాలుష్యంగా రూపాంతరం చెందుతుంది. దీని వల్ల మా చుట్టూ ప్రక్కల గ్రామాల ప్రజలు కష్టాలు ఎదుర్కోవాల్సిన పరిస్థితి వస్తుంది. కాబట్టి దానికి పరిష్కారం కనుక్కుందాము అన్న ఆలోచన నాకు వచ్చింది. దీన్ని మా టీచర్ గారు కూడా బాగుంది అన్నారు. నా ఆలోచన ఏంటంటే, గుళ్ళల్లో గాని ఇళ్ళల్లో గాని కావాల్సినన్ని టెంకాయ చిప్పలు దొరుకుతుంటాయి. వాటిల్లోని కొబ్బరిని వినియోగించిన తర్వాత ఆ ఆ కొబ్బరిచిప్పల్లో నారు నాటి మనము వాటిల్ని పెంచుకోవచ్చు. ప్లాస్టిక్ కవర్లను వాడేకన్నా ఈ కొబ్బరిచిప్పల్లో నారు నాటుకొని దాన్ని మొత్తాన్ని మనం భూమిలో నాటుకోవచ్చు లేదా మొక్క భూమిలో నాటేసిన తర్వాత ఆ చిప్పల్ని తిరిగి కొత్త నారు వేయటానికి ఉపయోగించుకోవచ్చు. ఇలా చేయడం వల్ల భూకాలుష్యాన్ని తగ్గించవచ్చు అని నా ఉద్దేశం.",
                "image": "This image captures three students presenting an eco-friendly innovation project, featuring saplings planted in biodegradable coconut shell containers, likely focused on sustainable agriculture or waste-to-wealth solutions for a school innovation competition.",
                "scores": {
                    "Novelty": {
                        "score": 6.0,
                        "reason": "Using coconut shells as biodegradable containers for saplings is a contextual adaptation, but similar natural alternatives to plastic covers are already known.",
                    },
                    "Usefulness": {
                        "score": 7.0,
                        "reason": "Directly reduces plastic use in sapling distribution by replacing covers with coconut shells, providing a practical alternative at the source.",
                    },
                    "Feasibility": {
                        "score": 7.0,
                        "reason": "Coconut shells are locally available and can hold soil and saplings; the prototype confirms basic usability but does not address durability or large-scale handling.",
                    },
                    "Scalability": {
                        "score": 7.0,
                        "reason": "Can be extended to farms and nurseries where coconut shells are available, though scaling depends on consistent shell supply and preparation",
                    },
                    "Sustainability": {
                        "score": 9.0,
                        "reason": "Replaces non-biodegradable plastic covers with natural coconut shells, reducing soil pollution and supporting environmentally friendly planting.",
                    },
                },
            },
            {
                "problem": "धरती पर सभी मनुष्य को जानवरों को जीवित रहने के लिए पानी की आवश्यकता है पर आजकल हम लोग पानी को बहुत ज्यादा पानी का दुरुपयोग कर रहे हैं जिससे सभी जानवर और मनुष्य को पानी उपलब्ध नहीं हो पता है तथा उन्हें बहुत गंदे पानी का उपयोग भी करना पड़ता है।सभी मनुष्यों जानवरों को जीवित रहने के लिए पीने के पानी की आवश्यकता है जिस प्रकार से आजकल हमारा विकास हो रहा है औद्योगीकरण हो रहा है जनसंख्या बढ़ रही है। उद्योग में बहुत पानी का इस्तेमाल होता है। कई जगह पर लोगों को पीने के लिए शुद्ध पानी भी उपलब्ध नहीं है कई स्थानों पर पानी बहुत गहराई में है और कई जगह अपनी जमीन के अंदर बहुत कम है वहां लोगों को बहुत दूर से पानी लाना पड़ रहा है।",
                "solution": """हम लोगों को पानी की इस समस्या से बचने के लिए बारिश के मौसम में बारिश का जो पानी जमीन पर गिरता है उसको हमारे घर में इकट्ठा करना चाहिए।
                बारिश के मौसम में बरसात का पानी जो हमारे छत पर गिरता है हमारे छत पर पाइप फिट करेंगे और पाइप को आंगन में बने कुंड या टांके में लगा देगे। 
                अब ध्यान देने वाली बात यह है कि हम उस पाईप में एक तो छत की तरफ वाले मुंह पर महीन छलनी का देगे उसे पाईप में ही लगा देगे ताकि छत का मोटा कचरा जैसे तिनके, कंकर पत्थर पाईप में ना आ सके।
                अब एक महीन छलनी टांके में जाने वाले पाईप के मुंह पर और लगाएंगे ताकि थोड़ी महीन कचरा भी पाईप में छनकर रह जाए और टांके में शूद्ध पानी जा सके।
                अब इस पानी का उपयोग हम खुद के पीने के पानी में, पशुओं को पिलाने में तथा खेत में भी कर सकते हैं।
                टांके में मोटर लगा सकते हैं जो बिजली से चले ताकि टांके से आसानी से पानी निकाला जा सके और इस पानी का उपयोग कर सके हम।
                मोटर को एक अलग पाईप से जोड़कर खेतो में पानी दे सकते हैंl
                इस प्रसार हम पानी को व्यर्थ होने से बचा स्केट हैं। और बारिश के पानी का सदुपयोग कर सकते हैं।""",
                "image": 'This drawing illustrates a rainwater harvesting system (labelled in Hindi as "वर्षा जल संग्रहण"), showing the process of collecting rainwater from a rooftop, filtering it, and storing or using it for purposes like garden irrigation.',
                "scores": {
                    "Novelty": {
                        "score": 3.0,
                        "reason": "Follows a standard rainwater harvesting model with no meaningful adaptation or original mechanism.",
                    },
                    "Usefulness": {
                        "score": 7.0,
                        "reason": "Effectively addresses water scarcity with a practical and relevant solution.",
                    },
                    "Feasibility": {
                        "score": 8.0,
                        "reason": "Uses commonly available components and describes a clear, implementable system.",
                    },
                    "Scalability": {
                        "score": 7.0,
                        "reason": "Replicable across regions with manageable adjustments, though installation requires moderate effort.",
                    },
                    "Sustainability": {
                        "score": 9.0,
                        "reason": "Reduces water wastage and promotes long-term use of a renewable resource.",
                    },
                },
            },
            {
                "problem": "பேட்ரி",
                "solution": "It was explained in our video",
                "scores": {
                    "Novelty": {
                        "score": 5.0,
                        "reason": "Arduino obstacle detection is a familiar kit-based concept with modest adaptation.",
                    },
                    "Usefulness": {
                        "score": 6.0,
                        "reason": "Real-time warning systems provide tangible safety improvement.",
                    },
                    "Feasibility": {
                        "score": 6.0,
                        "reason": "Prototype-level implementation is realistic, though railway deployment adds complexity.",
                    },
                    "Scalability": {
                        "score": 7.0,
                        "reason": "Sensor-based alert systems can be replicated across many rail environments.",
                    },
                    "Sustainability": {
                        "score": 6.0,
                        "reason": "The system supports long-term safety but depends on maintenance and integration.",
                    },
                },
            },
            {
                "problem": "सब्जियों व फलों के छिलकों का सड़ना उनका सही से निस्तारण न होना व उनके सड़ने से अनेक रोगों का होना",
                "solution": """*सब्जियों के छिलके से हैंडमेड पेपर तैयार करना*
                    *निर्माण विधि*- सब्जियों के छिलके विशेषकर आलू के छिलकों को मध्यम आंच पर रखकर उबाल लेंगे। छिलके इस प्रकार पानी में उबला जाए कि वह ना तो कच्चे रहे और ना ही ज्यादा गले। यदि छिलके ज्यादा गल जाएंगे तो पेपर नहीं बन पाएगा। अब छनने की सहायता से छिलकों से पानी अलग कर लिया जाएगा। इन छिलकों को समतल जगह जैसे मेंज, सेल्फ आदि पर फैला कर उस पर प्लास्टिक की शीट या पन्नी बिछा देंगे। प्लास्टिक शीट पर पर 3 -4 ईंटे या वजन रख देंगे। तीन-चार दिन बाद वजन हटाने पर हैंडमेड पेपर दिखेगा। जोकि छिलके से बना है। हम इस पेपर का उपयोग क्राफ्ट में कर सकते हैं। अलग-अलग सब्जी आलू प्याज के छिलके से अलग-अलग क्राफ्ट पेपर बना सकते हैं या कई सब्जियों के छिलके को साथ में मिलाकर एक हैंडमेड पेपर बना सकते हैं।
                    *उपयोग*- क्राफ्ट वर्क और ग्रीटिंग कार्ड बनाने में""",
                "image1": "Students participating in a sustainable waste management initiative by collecting fruit peels for disposal or potential composting.",
                "image2": "A project display board showcasing various creative crafts and utility items made by students using recycled and waste materials.",
                "image3": "A handwritten instructional document in Hindi explaining a zero-cost method for creating handmade paper from vegetable peels for craft applications.",
                "scores": {
                    "Novelty": {
                        "score": 5.0,
                        "reason": "Smart safety helmets are known, but student adaptation shows some originality.",
                    },
                    "Usefulness": {
                        "score": 6.0,
                        "reason": "Real-time hazard monitoring directly improves worker safety.",
                    },
                    "Feasibility": {
                        "score": 7.0,
                        "reason": "Sensor integration in wearables is practical with existing technology.",
                    },
                    "Scalability": {
                        "score": 7.0,
                        "reason": "The solution applies broadly across mining environments.",
                    },
                    "Sustainability": {
                        "score": 8.0,
                        "reason": "Durable safety equipment supports long-term operational use.",
                    },
                },
            },
            {
                "problem": "Power transmission lines are prone to faults such as short circuits, grounding issues, and fires. These faults can lead to power outages, equipment damage, and safety hazards for the public and workers",
                "solution": "We developed a Transmission Line Fault Detection System that monitors the power lines for faults such as short circuits, grounding, or fires. The system detects anomalies using sensors and immediately cuts power to the affected line to prevent further damage or accidents.\n How it works:\n oSensors are installed along the transmission line to detect abnormal conditions like high currents, grounding, or fires.\n oWhen a fault is detected, the system triggers a relay to isolate the faulty section.\n oA notification is sent to a central control system for further inspection and repair.\n Who it helps:\n oUtility companies by providing early fault detection and minimizing downtime.\n oResidents and businesses by reducing the duration of power outages.\n oPublic and workers by ensuring electrical safety.\n How it solves the problem:\n oPrevents escalation of faults into larger issues.\n oReduces response time for maintenance and repairs.\n oEnhances the safety and reliability of the power grid.",
                "scores": {
                    "Novelty": {
                        "score": 4.0,
                        "reason": "Fault detection systems are standard infrastructure solutions with minor student framing.",
                    },
                    "Usefulness": {
                        "score": 5.0,
                        "reason": "Early detection meaningfully reduces outage risk.",
                    },
                    "Feasibility": {
                        "score": 6.0,
                        "reason": "Sensor + relay systems are implementable with moderate complexity.",
                    },
                    "Scalability": {
                        "score": 7.0,
                        "reason": "Infrastructure monitoring scales across grid networks.",
                    },
                    "Sustainability": {
                        "score": 5.0,
                        "reason": "Requires continuous upkeep and resource investment.",
                    },
                },
            },
            {
                "problem": "1.Data privacy: Issues with data privacy, phishing, and data mixups.\n 2. Remote internet speed: Issues with remote internet speed and connections.\n 3. Deepfake content: Deepfake content is a major tech problem.\n 4. Remembering passwords: Many people have trouble remembering online passwords.\n 5. Mobile security: Mobile security is a major tech problem.",
                "solution": "On Disscusing with iliterate and Digitally iliterate people my team came to the solution that before using technology in any type must aware about the pro's and con's of the Technology . My teams solution works in this way that every user must Install antivirus software. Download apps from trusted sources. also arrange a training program for those people's who don't know about the technology and also different organisations need to organise awareness campaign or workshop in rural areas where it needs most and make them digitally literate atleast they didn't become victim of fraud. I think this's need of hour that we open digital awareness classrooms where people get trained in every corner of the country. And the person who sit in these class rooms helps in any kind of misshapping in digital world e. g( Deepfake, phishing etc.) In this way we can solve these problems.",
                "scores": {
                    "Novelty": {
                        "score": 2.0,
                        "reason": "Awareness campaigns and antivirus advice are common digital literacy solutions.",
                    },
                    "Usefulness": {
                        "score": 3.0,
                        "reason": "Provides some social benefit but lacks concrete system design.",
                    },
                    "Feasibility": {
                        "score": 7.0,
                        "reason": "Education and awareness programs are highly achievable.",
                    },
                    "Scalability": {
                        "score": 5.0,
                        "reason": "Expansion is possible but requires organizational effort.",
                    },
                    "Sustainability": {
                        "score": 5.0,
                        "reason": "Programs can continue with funding but demand ongoing input.",
                    },
                },
            },
            {
                "problem": "ARTIFICIAL OR CHEMICAL FERTILIZERS ARE NOT REQUIRED",
                "solution": "AQUAPONICS IS A TECHNIQUE WHICH IS COMBINATION OF AQUACULTURE(growing fish)AND HYDROPONICS(growing plants in non soil media and nutrition laden water).AQUAPONICS MAY ALSO PREVENT TOXIC WATER RUNOFF.IT ALSO MAINTAINS ECOSYSTEM BALANCE BY RECYCLING THE WASTE AND EXCREATORY PRODUCTS PRODUCED BY THE FISH",
                "scores": {
                    "Novelty": {
                        "score": 4.0,
                        "reason": "Aquaponics is a known sustainability technique without novel adaptation described.",
                    },
                    "Usefulness": {
                        "score": 7.0,
                        "reason": "Reduces fertilizer dependence with real environmental benefit.",
                    },
                    "Feasibility": {
                        "score": 8.0,
                        "reason": "Small-scale aquaponic systems are practical to implement.",
                    },
                    "Scalability": {
                        "score": 9.0,
                        "reason": "Systems can expand from household to commercial levels.",
                    },
                    "Sustainability": {
                        "score": 9.0,
                        "reason": "Closed-loop nutrient cycling supports long-term ecological balance.",
                    },
                },
            },
        ],
    }


def inference(
    *,
    problem: str,
    solution: str,
    image: str | list[str] | None = None,
    file: str | list[str] | None = None,
    provider_name: Literal["openai", "google"] = "openai",
    model_name: str = "gpt-4o-mini",
) -> str:
    solution_schema = get_evaluation_system_schema()
    examples_dataset = get_evaluation_examples()

    instructions = (
        "# Evaluation System Schema\n"
        f"{json.dumps(solution_schema, indent=2)}\n\n"
        "# Few-Shot Examples\n"
        f"{json.dumps(examples_dataset, indent=2)}\n\n"
        "# Output should be strictly in json format"
    )

    completion_config = KaapiCompletionConfig(
        provider=provider_name,
        type="text",
        params=TextLLMParams(
            model=model_name, instructions=instructions, response_format="json_object"
        ).model_dump(),
    )

    completion_config, warnings = transform_kaapi_config_to_native(completion_config)

    # configuration
    configuration = LLMCallConfig(blob=ConfigBlob(completion=completion_config))

    inputs = [
        {
            "type": "text",
            "content": {
                "format": "text",
                "value": f"# Problem Statement\n:{problem}\n\n# Solution to the problem statement:\n{solution}\n\n Respond in json format.",
            },
        }
    ]
    # build image inputs
    if image is not None:
        images = [image] if isinstance(image, str) else image
        for img in images:
            inputs.append({"type": "image", "content": {"format": "url", "value": img}})

    query = QueryParams(input=inputs)

    # build llm call contract
    request = LLMCallRequest(query=query, config=configuration)

    provider_name = request.config.blob.completion.provider
    provider_class = LLMProvider.get_provider_class(provider_type=provider_name)
    credential = {"api_key": os.getenv("OPENAI_API_KEY")}
    client = provider_class.create_client(credentials=credential)
    provider_instance = provider_class(client=client)

    response: LLMCallResponse | None
    with resolved_input_context(query_input=request.query.input) as resolved_input:
        response, error = provider_instance.execute(
            completion_config=request.config.blob.completion,
            query=query,
            resolved_input=resolved_input,
            include_provider_raw_response=False,
        )

    return response.response.output.content.value


def run_inference_test_cases():
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "output_multi/inference")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    outputs = []

    # CASE 1: PROBLEM SOLUTION TEXT INPUT
    problem = """Project synopsis :- In many places, many goods are stolen from the grid like Copper plates, wires, aluminium and these goods are very expensive due to which there is a lot of loss. Dispute there has been a 24 hour security guard there, but theft happens because one security guard cannot Guard everywhere . Every item like Copper plates, wires etc are expensive and sometimes electricity is also stolen.""".strip()
    solution = """
    We developed a system in which we will place mirrors in such a way that they can reflect the light and also place a laser light at one corner and also place a receiver in which there is an alarm which sounds when light is not received .

    Because one security guard cannot Guard everywhere so, when we plays this system automatically when someone come inside from some other ways the alarm will sound and guard been alert .(https://youtu.be/gKJiCGNaAwg?si=xMmwyxQNDkwS7kD5)
    """.strip()

    response_case1 = inference(problem=problem, solution=solution)
    outputs.append(
        f"Problem: {problem}\n"
        f"Solution: {solution}\n\n"
        "------- CASE 1: PROBLEM SOLUTION TEXT INPUT  ---------\n"
        + str(response_case1)
        + "\n\n"
    )

    # CASE 2: PROBLEM SOLUTION STATEMENT WITH IMAGE
    problem = """Project synopsis :- In many places, many goods are stolen from the grid like Copper plates, wires, aluminium and these goods are very expensive due to which there is a lot of loss. Dispute there has been a 24 hour security guard there, but theft happens because one security guard cannot Guard everywhere . Every item like Copper plates, wires etc are expensive and sometimes electricity is also stolen.""".strip()
    solution = """
    We developed a system in which we will place mirrors in such a way that they can reflect the light and also place a laser light at one corner and also place a receiver in which there is an alarm which sounds when light is not received .

    Because one security guard cannot Guard everywhere so, when we plays this system automatically when someone come inside from some other ways the alarm will sound and guard been alert .(https://youtu.be/gKJiCGNaAwg?si=xMmwyxQNDkwS7kD5)
    """.strip()
    image = "https://drive.usercontent.google.com/download?id=1Xa-_KlHb9tyFLNcFXRKQ4KylVf5kGa4c&export=download&authuser=0"

    response_case2 = inference(problem=problem, solution=solution, image=image)
    outputs.append(
        f"Image URL: {image}\n"
        "------- CASE 2: PROBLEM SOLUTION STATEMENT WITH IMAGE  ---------\n"
        + str(response_case2)
        + "\n\n----------------------------\n\n"
    )

    # CASE 3: MULTI-LINGUAL PROBLEM SOLUTION STATEMENT
    problem = """సాధారణంగా మనం చూసినట్లయితే మన ఇండ్లలో సీలింగ్లలో ఏర్పడే బల్లి పాతరను వెబ్ రిమూవర్ స్టిక్ తో తొలగించినప్పుడు ఆ బల్లి పాతర మన కళ్ళల్లో పడుతుంది. అలాగే చర్మం మీద పడి దురద లేస్తుంది. అంతేకాకుండా ఎక్కువ సమయం పాటు మనము తీయడానికి ప్రయత్నించినప్పుడు చేతులు కూడా నొప్పులు పెడతాయి.""".strip()
    solution = """
    women friendly web remover stick.
    సాధారణంగా మనం చూసినట్లయితే మన ఇండ్లలో సీలింగ్లలో ఏర్పడే బల్లి పాతరను వెబ్ రిమూవర్ స్టిక్ తో తొలగించినప్పుడు ఆ బల్లి పాతర మన కళ్ళల్లో పడుతుంది. అలాగే చర్మం మీద పడి దురద లేస్తుంది. అంతేకాకుండా ఎక్కువ సమయం పాటు మనము తీయడానికి ప్రయత్నించినప్పుడు చేతులు కూడా నొప్పులు పెడతాయి. ఈ సమస్యకు పరిష్కారంగా మేము ఉమెన్ ఫ్రెండ్లీ వెబ్ రిమూవర్ స్టిక్ తయారు చేసాము. మోటార్ సహాయంతో వేబ్ రిమూవర్ తిరుగుతుంది ఇది తిరిగినప్పుడు పడే బల్లి పాతర అంతా మన మీద పడకుండా మధ్యలో ఒక అంబ్రెల్లా ఉంచడం జరిగింది. ఈ అంబ్రెల్లా బల్లి పాతరను కలెక్ట్ చేసుకుంటుంది అంటే మన మీద పడకుండా చూస్తుంది. దీని ఈజీగా తక్కువ ఖర్చుతో తయారు చేయవచ్చు సులభంగా వినియోగించవచ్చు.
    """.strip()

    response_case3 = inference(problem=problem, solution=solution)
    outputs.append(
        f"Problem: {problem}\n"
        f"Solution: {solution}\n"
        "------- CASE 3: MULTI-LINGUAL PROBLEM SOLUTION STATEMENT  ---------\n"
        + str(response_case3)
        + "\n\n"
    )

    # CASE 4: MULTI-LINGUAL PROBLEM SOLUTION STATEMENT
    problem = """సాధారణంగా మనం చూసినట్లయితే మన ఇండ్లలో సీలింగ్లలో ఏర్పడే బల్లి పాతరను వెబ్ రిమూవర్ స్టిక్ తో తొలగించినప్పుడు ఆ బల్లి పాతర మన కళ్ళల్లో పడుతుంది. అలాగే చర్మం మీద పడి దురద లేస్తుంది. అంతేకాకుండా ఎక్కువ సమయం పాటు మనము తీయడానికి ప్రయత్నించినప్పుడు చేతులు కూడా నొప్పులు పెడతాయి.""".strip()
    solution = """
    women friendly web remover stick.
    సాధారణంగా మనం చూసినట్లయితే మన ఇండ్లలో సీలింగ్లలో ఏర్పడే బల్లి పాతరను వెబ్ రిమూవర్ స్టిక్ తో తొలగించినప్పుడు ఆ బల్లి పాతర మన కళ్ళల్లో పడుతుంది. అలాగే చర్మం మీద పడి దురద లేస్తుంది. అంతేకాకుండా ఎక్కువ సమయం పాటు మనము తీయడానికి ప్రయత్నించినప్పుడు చేతులు కూడా నొప్పులు పెడతాయి. ఈ సమస్యకు పరిష్కారంగా మేము ఉమెన్ ఫ్రెండ్లీ వెబ్ రిమూవర్ స్టిక్ తయారు చేసాము. మోటార్ సహాయంతో వేబ్ రిమూవర్ తిరుగుతుంది ఇది తిరిగినప్పుడు పడే బల్లి పాతర అంతా మన మీద పడకుండా మధ్యలో ఒక అంబ్రెల్లా ఉంచడం జరిగింది. ఈ అంబ్రెల్లా బల్లి పాతరను కలెక్ట్ చేసుకుంటుంది అంటే మన మీద పడకుండా చూస్తుంది. దీని ఈజీగా తక్కువ ఖర్చుతో తయారు చేయవచ్చు సులభంగా వినియోగించవచ్చు.
    """.strip()

    image = "https://drive.usercontent.google.com/download?id=1gVrcB4Jy3zjMBxRStS9NBMksvXdjcKnc&export=download&authuser=0"

    response_case4 = inference(problem=problem, solution=solution, image=image)
    outputs.append(
        f"Image URL: {image}\n\n"
        "------- CASE 4: MULTI-LINGUAL PROBLEM SOLUTION STATEMENT WITH IMAGE  ---------\n"
        + str(response_case4)
    )

    output_file = os.path.join(OUTPUT_FOLDER, "test_cases_output.txt")
    with open(output_file, "w") as f:
        f.write("".join(outputs))


### ----------------------  ITERATION BASED RESPONSE ------------------------------------------
REQUIRED_COLUMNS = {"problem", "solution"}


def _read_and_validate_file(file_path: Path) -> pd.DataFrame:
    ext = file_path.suffix.lower()
    file_data = None
    if ext == ".csv":
        file_data = pd.read_csv(file_path)
    elif ext in {".xlsx", ".xls"}:
        file_data = pd.read_excel(file_path)

    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .csv, .xlsx, or .xls")

    file_data = file_data.dropna(how="all")
    file_data.columns = file_data.columns.str.lower().str.strip()
    missing = REQUIRED_COLUMNS - set(file_data.keys())
    if missing:
        raise ValueError(f"Input dict must have both 'problem' and 'solution' keys")

    return file_data


def _validate_dict_input(input_data: List[dict[str, str]]) -> None:
    for item in input_data:
        missing = REQUIRED_COLUMNS - set(item.keys())
        if missing:
            raise ValueError("Input dict must have both 'problem' and 'solution' keys")
        if not item.get("problem") or not item.get("solution"):
            raise ValueError(
                "'problem' and 'solution' values must be non-empty strings"
            )


def _generate_id(length=5):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


class EvaluationInput(BaseModel):
    problem: str
    solution: str
    custom_id: str | None
    image: str | List[str] | None

    @field_validator("problem", "solution")
    @classmethod
    def must_be_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Value must be a non-empty string")
        return v.strip()

def _to_direct_url(url: str) -> str:
    """Convert Google Drive sharing links to direct image URLs."""
    import re
    url = url.strip()

    file_id = None

    # https://drive.google.com/file/d/{ID}/...
    m = re.match(r"https://drive\.google\.com/file/d/([^/]+)", url)
    if m:
        file_id = m.group(1)

    # https://drive.google.com/open?id={ID}
    if not file_id:
        m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
        if m and "drive.google.com" in url:
            file_id = m.group(1)

    # https://drive.usercontent.google.com/download?id={ID}
    if not file_id:
        m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
        if m and "drive.usercontent.google.com" in url:
            file_id = m.group(1)

    if file_id:
        return f"https://lh3.googleusercontent.com/d/{file_id}"

    return url

def run_inference_batch(
    *,
    input: Path | List[dict[str, str]],
    output_path: str,
    provider_name: Literal["openai", "google"] = "openai",
    model_name: str = "gpt-4o-mini",
    run_test_images: bool = False
) -> None:

    data: List[dict[str, str]] | None = None

    if run_test_images:
        print(f"\n\n------------------RUNNING MODEL: {provider_name}/{model_name} WITH IMAGES ------------------\n\n")
    else:
        print(f"\n\n------------------RUNNING MODEL: {provider_name}/{model_name} W/O IMAGES ------------------\n\n")

    if isinstance(input, Path):
        if os.path.exists(input):
            data = _read_and_validate_file(input)

            data = [
                EvaluationInput(
                    problem=row["problem"],
                    solution=row["solution"],
                    custom_id=str(row["cid"]) if pd.notna(row.get("cid")) else None,
                    image=str(row["documents"]) if pd.notna(row.get("documents")) else None,
                )
                for _, row in data.iterrows()
                if pd.notna(row["problem"]) and pd.notna(row["solution"])
            ]
        else:
            raise ValueError(f"File doesn't exist : {input}")

    elif isinstance(input, List):
        _validate_dict_input(input_data=input)
        data = [
            EvaluationInput(
                problem=item["problem"],
                solution=item["solution"],
                image=item["image"] if item.get("image") else None,
            )
            for item in input
            if item["problem"] and item["solution"]
        ]
    else:
        raise ValueError("make sure the input is either path or valid dict")

    if data:
        # creating batch jsonl
        # step 1: create configuration
        solution_schema = get_evaluation_system_schema()
        examples_dataset = get_evaluation_examples()

        instructions = (
            "# Evaluation System Schema\n"
            f"{json.dumps(solution_schema, indent=2)}\n\n"
            "# Few-Shot Examples\n"
            f"{json.dumps(examples_dataset, indent=2)}\n\n"
            "# Output should be strictly in json format"
        )

        output = []
        completion_config = KaapiCompletionConfig(
            provider=provider_name,
            type="text",
            params=TextLLMParams(
                model=model_name,
                instructions=instructions,
                temperature=0.4,
                output_schema={
                    "type": "object",
                    "properties": {
                       
                        "Novelty": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "number"},
                                "reason": {"type": "string"},
                            },
                            "required": ["score", "reason"],
                            "additionalProperties": False,
                        },
                        "Usefulness": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "number"},
                                "reason": {"type": "string"},
                            },
                            "required": ["score", "reason"],
                            "additionalProperties": False,
                        },
                        "Feasibility": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "number"},
                                "reason": {"type": "string"},
                            },
                            "required": ["score", "reason"],
                            "additionalProperties": False,
                        },
                        "Scalability": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "number"},
                                "reason": {"type": "string"},
                            },
                            "required": ["score", "reason"],
                            "additionalProperties": False,
                        },
                        "Sustainability": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "number"},
                                "reason": {"type": "string"},
                            },
                            "required": ["score", "reason"],
                            "additionalProperties": False,
                        },
                        **({"Image_contains_summary": {"type": "string"}} if run_test_images else {}),
                    },
                    "required": [
                        "Novelty",
                        "Usefulness",
                        "Feasibility",
                        "Scalability",
                        "Sustainability",
                        *(["Image_contains_summary"] if run_test_images else []),
                    ],
                    "additionalProperties": False,
                },
            ).model_dump(exclude_none=True),
        )

        completion_config, warnings = transform_kaapi_config_to_native(
            completion_config
        )

        configuration = LLMCallConfig(blob=ConfigBlob(completion=completion_config))
        provider_class = LLMProvider.get_provider_class(provider_type=provider_name)

        if provider_name == "openai":
            credential = {"api_key": os.getenv("OPENAI_API_KEY")}
        else:
            credential = {"api_key": os.getenv("GEMINI_API_KEY")}
        client = provider_class.create_client(credentials=credential)
        provider_instance = provider_class(client=client)
        response: LLMCallResponse | None

        for input_dict in data:
            input_dict = input_dict.model_dump()
            custom_id = (
                input_dict.get("custom_id") or f"student_{_generate_id(length=6)}"
            )
            problem = input_dict.get("problem")
            solution = input_dict.get("solution")
            image = input_dict.get("image", None) if run_test_images else None


            if image is not None:
                if isinstance(image, str):
                    image = _to_direct_url(image)
                elif isinstance(image, list):
                    image = [_to_direct_url(img) for img in image]

            inputs = [
                {
                    "type": "text",
                    "content": {
                        "format": "text",
                        "value": f"# Problem Statement\n:{problem}\n\n# Solution to the problem statement:\n{solution}\n\n Respond in json format.",
                    },
                }
            ]

            if image is not None:
                images = [image] if isinstance(image, str) else image
                for img in images:
                    inputs.append({"type": "image", "content": {"format": "url", "value": img}})

            query = QueryParams(input=inputs)

            request = LLMCallRequest(query=query, config=configuration)

            with resolved_input_context(query_input=request.query.input) as resolved_input:
                response, error = provider_instance.execute(
                    completion_config=request.config.blob.completion,
                    query=query,
                    resolved_input=resolved_input,
                    include_provider_raw_response=False,
                )

            if error is not None or response is None:
                logger.error(f"[run_inference_batch] Error for CID {custom_id}: {error}")
                response_output = None
            else:
                try:
                    response_output = json.loads(response.response.output.content.value)
                except json.JSONDecodeError as e:
                    logger.error(f"[run_inference_batch] JSON decode error for CID {custom_id}: {e}")
                    response_output = None

            if run_test_images:
                output.append(
                    {
                        "CID": custom_id,
                        "Problem": problem,
                        "Solution": solution,
                        "Image URL": image,
                        "scores_with_images": response_output,
                    }
                )
            else:
                output.append(
                    {
                        "scores_without_images": response_output,
                    }
                )

        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(output, f, indent=2)

    else:
        raise ValueError("batch id didn't got")


if __name__ == "__main__":
    # TODO: uncomment to run inference test cases
    #run_inference_test_cases()

    # run_inference_batch(Path(os.path.join(BASE_DIR, "mutli_input_batch.xlsx")))

    run_test_images = False

    provider_name = "google"
    model_name = "gemini-3.1-flash-lite-preview"

    if run_test_images:
        OUTPUT_FOLDER = os.path.join(BASE_DIR, f"output_multiple_imagesv1/{provider_name}/{model_name}")
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        OUTPUT_FILE = f"{model_name}.json"
        OUTPUT_FILE_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)
    else:
        OUTPUT_FOLDER = os.path.join(BASE_DIR, f"output_multiple_imagesv1/{provider_name}/{model_name}-wo-images")
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        OUTPUT_FILE = f"{model_name}-wo-images.json"
        OUTPUT_FILE_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)

    run_inference_batch(
        input=Path(os.path.join(BASE_DIR, "200 Golden_dataset_2.O-3.xlsx")),
        output_path=OUTPUT_FILE_PATH,
        provider_name=provider_name,
        model_name=model_name,
        run_test_images=run_test_images
    )
"""
Score-Only Evaluation Script — Scores student submissions across 5 dimensions.

HOW TO USE:
1. Scroll down to the bottom of this file (if __name__ == "__main__")
2. Set PROVIDER, MODEL, INPUT_FILE, OUTPUT_PATH, USE_IMAGES, TEMPERATURE
3. Run:  uv run -m app.inquilab.run_evaluation_score
4. Output goes to the OUTPUT_PATH you specify
"""

from typing import List
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

import random
import json
import string
import pandas as pd
from pathlib import Path
from app.services.llm.mappers import transform_kaapi_config_to_native
from logging import Logger
import re

load_dotenv()

logger = Logger(__name__)

METRICS = ["Novelty", "Usefulness", "Feasibility", "Scalability", "Sustainability"]
BASE_DIR = Path(__file__).parent


# =====================================================================
#  PROMPTS — Edit these functions to change AI behavior
# =====================================================================


def get_system_role() -> str:
    """
    WHO is the AI pretending to be?
    Edit this to change the evaluator's persona and background.
    """
    return """You are a senior innovation evaluator for the School Innovation Marathon (SIM) - India's largest student innovation platform.
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
    2. Rule 2: Prototype Impact on Scoring : Prototype evidence should only change scores if it introduces new functional insight or reduces uncertainty about how the solution works.
    3. Rule 3: Prototype vs Effort : A prototype demonstrates student effort and experimentation, which should be acknowledged in evaluation, but effort alone should not lead to higher scores without additional functional evidence.
    4. Rule 4: Irrelevant Evidence Handling : If additional images or documents are not clearly related to the problem or solution, they must be ignored and should not influence scoring.
    5. Rule 5: Generic Solution without Explanation : If a student suggests a widely known solution without explaining how it works or how they will implement it, assign: Low novelty (1-2), Mid-range feasibility and scalability (6-7).
    6. Rule 6: Prototype Adds New Design Evidence -> Scores Increase.
    7. Rule 7: Prototype Only Confirms Existing Understanding -> Scores Do Not Increase.
    8. Rule 8: Design Clarity vs Technical Depth (Score Ceiling) : Even when a prototype adds clear design understanding, if it does not explain deeper aspects such as why the design works better, performance, durability, or optimization, scores should not reach the highest range (9-10).

CRITICAL SCORING GUARDRAILS:
- Scores MUST be justified by explicit evidence in the submission.
- Do NOT assume resources, infrastructure, or execution capability unless stated.
- If the idea and problem donot relate to each other then score least to all parameters.
- Generic textbook or widely known ideas must be scored low on novelty unless clear original adaptation is shown.
- If the model cannot clearly interpret the attachment, do not fabricate details. Score based on available text evidence only.

HUMAN EVALUATION PRINCIPLES:
- Recognize visible student effort and original thinking even if execution is incomplete
- Reward contextual relevance and real-world problem awareness
- Separate sustainability intention from operational sustainability
- Avoid over-penalizing grammar or presentation
- Do not inflate scores without evidence

When evidence is partial:
-> score conservatively but acknowledge effort
-> use mid-range scores when student reasoning is present but incomplete"""


def get_evaluation_objective() -> str:
    """
    WHAT should the AI evaluate?
    Edit this to change the evaluation goals.
    """
    return """Evaluate student innovation submissions to identify genuinely promising ideas that demonstrate:
1. Real problem understanding and original thinking
2. Practical solutions that could realistically be implemented
3. Potential for meaningful impact in the Indian context
4. Consideration of sustainability and scalability factors
5. Evidence of student's own creative thinking vs. copied ideas

Distinguish clearly between:
- intention vs execution detail
- concept vs deployable solution
- originality vs adaptation

Your evaluation directly impacts which students advance in India's premier innovation program, so accuracy and fairness are paramount."""


def get_evaluation_method() -> str:
    """
    HOW should the AI evaluate?
    Edit this to change the step-by-step evaluation process.
    """
    return """EVALUATION PROCESS:

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
    c. Evidence from Attachments - Interpret the attachments to identify problem , Idea , Solution design/model etc for supporting evidences if any.

STEP 3: EVIDENCE-BASED SCORING
- Score each parameter 1-10 based on specific evidence in the submission
- Provide one clear, concise reason for each score focusing on what you observed
- Be consistent: similar quality submissions should receive similar scores
- Be fair: don't penalize for age-appropriate language or presentation style
- Assess real-world execution factors: cost, infrastructure, technical complexity, maintenance

STEP 4: OUTPUT GENERATION
- Return ONLY valid JSON with exact structure specified
- Each reason should be one clear sentence explaining the score
- Ensure scores align with the detailed rubrics provided"""


def get_scoring_criteria() -> list:
    """
    SCORING RUBRICS — What scores mean for each criterion.
    Edit the descriptions or scale to change how scores are assigned.
    """
    return [
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
            "scoring_notes": "Known textbook projects, kit-based builds, or copied concepts must score 1-4 unless meaningful innovation is demonstrated.",
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
    ]


def get_few_shot_examples():
    """
    EXAMPLES — These teach the AI what good evaluations look like.
    Edit, add, or remove examples to steer the AI's scoring behavior.
    Each example has: problem, solution, (optional) image description, and expected scores.
    """
    return [
        {
            "problem": "Nowadays thefts are very much in the society. They are robbering the homes, offices, banks etc. So controlling theft in the society.",
            "solution": "Anti theft alram is a sensor based it detects the when theft is going to theft the materials in home iffice also in a banks.",
            "image": "The image depicts a student-made prototype of a model house, likely demonstrating a project on smart home automation, energy efficiency, or electrical circuitry.",
            "scores": {
                "Novelty": {"score": 2.0, "reason": "Proposes a sensor-based alarm for detecting theft, which is already a widely used approach in home and building security, with no new mechanism or adaptation described."},
                "Usefulness": {"score": 3.0, "reason": "Targets a real problem (theft), but does not explain how the alarm detects intrusion, who gets alerted, or how it helps stop or respond to theft in practice."},
                "Feasibility": {"score": 4.0, "reason": "A basic alarm system using sensors and wiring is implementable, and the prototype shows a wired setup with a light, but absence of sensor type, detection logic, and alert system reduces confidence in execution."},
                "Scalability": {"score": 3.0, "reason": "Although intended for homes, offices, and banks, these environments require different levels of security; the solution does not specify how the same alarm system would adapt across these contexts."},
                "Sustainability": {"score": 6.0, "reason": "Electronic alarm systems are generally low-maintenance once installed, but the solution does not provide details on power usage, durability, or long-term operation"},
            },
        },
        {
            "problem": "సరుగుడు బొప్పాయి వంటి తోటలు వేసుకునే వాళ్ళు నారుని చిన్నచిన్న నల్లటి ప్లాస్టిక్ సంచుల్లో తెచ్చుకుంటారు అలాగే ఇళ్లల్లో కొన్ని రకాల పూల చెట్లు కానివ్వండి పండ్ల చెట్లు కానివ్వండి వేసుకోవాలనుకున్నప్పుడు కూడా వారు ఆ చెట్లని నల్లటి ప్లాస్టిక్ సంచుల్లో ఇవ్వడం అనేది మనం చూస్తుంటాం ఈ సంచులు తర్వాత ఉపయోగపడం రీసైకిలింగ్ చేయలేం అలాగే ఇవి అలాగే గనక పడేస్తే ప్లాస్టిక్ కాలుష్యం కింద భూమిని కలుషితం చేస్తాయి కనుక ఈ ఎక్కువ చెట్లని తెచ్చుకునేటువంటి రైతులు ఎక్కువగా ఈ ప్లాస్టిక్ కవర్లను ఉపయోగించటం అనేది నేను గమనించినటువంటి పెద్ద సమస్య ఇది భూ కాలుష్యానికి సంబంధించిన సమస్య ఇల్లా ఇళ్లల్లోనూ ఇది సమస్యగానే ఉంది తోటల్లోనూ ఇది సమస్యగానే నాకు కనిపించింది కాబట్టి దీని పరిష్కరించాలని చిన్న ప్రయత్నాన్ని చేయడం జరిగింది",
            "solution": "రైతులు పండ్ల చెట్లు, నిమ్మనారు ఇలాంటివన్నీ తెచ్చుకునేటప్పుడు చిన్న చిన్న ప్లాస్టిక్ కవర్లలో వాళ్ళు నారుని తెచ్చుకుంటూ ఉంటారు. అది రీసైక్లింగ్ కి పనికిరాదు కాబట్టి వాళ్ళు ఆ ప్లాస్టిక్ ని బయటే పారేస్తారు. ప్లాస్టిక్ కాబట్టి అది నీటిని భూమిలోకి చేరనివ్వదు, ఆ ప్లాస్టిక్ భూమిలో కరిగిపోకుండా కొన్ని లక్షల సంవత్సరాలు భూమిలోపల ఉండి భూకాలుష్యంగా రూపాంతరం చెందుతుంది. దీని వల్ల మా చుట్టూ ప్రక్కల గ్రామాల ప్రజలు కష్టాలు ఎదుర్కోవాల్సిన పరిస్థితి వస్తుంది. కాబట్టి దానికి పరిష్కారం కనుక్కుందాము అన్న ఆలోచన నాకు వచ్చింది. దీన్ని మా టీచర్ గారు కూడా బాగుంది అన్నారు. నా ఆలోచన ఏంటంటే, గుళ్ళల్లో గాని ఇళ్ళల్లో గాని కావాల్సినన్ని టెంకాయ చిప్పలు దొరుకుతుంటాయి. వాటిల్లోని కొబ్బరిని వినియోగించిన తర్వాత ఆ ఆ కొబ్బరిచిప్పల్లో నారు నాటి మనము వాటిల్ని పెంచుకోవచ్చు. ప్లాస్టిక్ కవర్లను వాడేకన్నా ఈ కొబ్బరిచిప్పల్లో నారు నాటుకొని దాన్ని మొత్తాన్ని మనం భూమిలో నాటుకోవచ్చు లేదా మొక్క భూమిలో నాటేసిన తర్వాత ఆ చిప్పల్ని తిరిగి కొత్త నారు వేయటానికి ఉపయోగించుకోవచ్చు. ఇలా చేయడం వల్ల భూకాలుష్యాన్ని తగ్గించవచ్చు అని నా ఉద్దేశం.",
            "image": "These students are addressing the environmental problem of plastic waste in plant nurseries by promoting an eco-friendly solution: using biodegradable coconut shells as sustainable alternatives to plastic seedling bags.",
            "scores": {
                "Novelty": {"score": 6.0, "reason": "Using coconut shells as biodegradable containers for saplings is a contextual adaptation, but similar natural alternatives to plastic covers are already known."},
                "Usefulness": {"score": 7.0, "reason": "Directly reduces plastic use in sapling distribution by replacing covers with coconut shells, providing a practical alternative at the source."},
                "Feasibility": {"score": 7.0, "reason": "Coconut shells are locally available and can hold soil and saplings; the prototype confirms basic usability but does not address durability or large-scale handling."},
                "Scalability": {"score": 7.0, "reason": "Can be extended to farms and nurseries where coconut shells are available, though scaling depends on consistent shell supply and preparation."},
                "Sustainability": {"score": 9.0, "reason": "Replaces non-biodegradable plastic covers with natural coconut shells, reducing soil pollution and supporting environmentally friendly planting."},
            },
        },
        {
            "problem": """धरती पर सभी मनुष्य को जानवरों को जीवित रहने के लिए पानी की आवश्यकता है पर आजकल हम लोग पानी को बहुत ज्यादा पानी का दुरुपयोग कर रहे हैं जिससे सभी जानवर और मनुष्य को पानी उपलब्ध नहीं हो पता है तथा उन्हें बहुत गंदे पानी का उपयोग भी करना पड़ता है।सभी मनुष्यों जानवरों को जीवित रहने के लिए पीने के पानी की आवश्यकता है जिस प्रकार से आजकल हमारा विकास हो रहा है औद्योगीकरण हो रहा है जनसंख्या बढ़ रही है।
                    उद्योग में बहुत पानी का इस्तेमाल होता है। कई जगह पर लोगों को पीने के लिए शुद्ध पानी भी उपलब्ध नहीं है कई स्थानों पर पानी बहुत गहराई में है और कई जगह अपनी जमीन के अंदर बहुत कम है वहां लोगों को बहुत दूर से पानी लाना पड़ रहा है।""",
            "solution": """ हम लोगों को पानी की इस समस्या से बचने के लिए बारिश के मौसम में बारिश का जो पानी जमीन पर गिरता है उसको हमारे घर में इकट्ठा करना चाहिए।
                बारिश के मौसम में बरसात का पानी जो हमारे छत पर गिरता है हमारे छत पर पाइप फिट करेंगे और पाइप को आंगन में बने कुंड या टांके में लगा देगे। 
                अब ध्यान देने वाली बात यह है कि हम उस पाईप में एक तो छत की तरफ वाले मुंह पर महीन छलनी का देगे उसे पाईप में ही लगा देगे ताकि छत का मोटा कचरा जैसे तिनके, कंकर पत्थर पाईप में ना आ सके।
                अब एक महीन छलनी टांके में जाने वाले पाईप के मुंह पर और लगाएंगे ताकि थोड़ी महीन कचरा भी पाईप में छनकर रह जाए और टांके में शूद्ध पानी जा सके।
                अब इस पानी का उपयोग हम खुद के पीने के पानी में, पशुओं को पिलाने में तथा खेत में भी कर सकते हैं।
                टांके में मोटर लगा सकते हैं जो बिजली से चले ताकि टांके से आसानी से पानी निकाला जा सके और इस पानी का उपयोग कर सके हम।
                मोटर को एक अलग पाईप से जोड़कर खेतो में पानी दे सकते हैंl
                इस प्रसार हम पानी को व्यर्थ होने से बचा स्केट हैं। और बारिश के पानी का सदुपयोग कर सकते हैं।""",
            "image": "This rainwater harvesting (varsha jal sangrahan) project addresses water scarcity by collecting and filtering runoff from rooftops to provide a sustainable irrigation source for crops.",
            "scores": {
                "Novelty": {"score": 3.0, "reason": "Follows a standard rainwater harvesting model with no meaningful adaptation or original mechanism"},
                "Usefulness": {"score": 7.0, "reason": "Effectively addresses water scarcity with a practical and relevant solution."},
                "Feasibility": {"score": 8.0, "reason": "Uses commonly available components and describes a clear, implementable system."},
                "Scalability": {"score": 7.0, "reason": "Replicable across regions with manageable adjustments, though installation requires moderate effort."},
                "Sustainability": {"score": 9.0, "reason": "Reduces water wastage and promotes long-term use of a renewable resource."},
            },
        },
        
    ]


def get_user_prompt(problem: str, solution: str) -> str:
    """
    USER PROMPT — This is sent for each submission.
    Edit this to change what the AI sees for each student's work.
    """
    return (
        f"# Problem Statement\n:{problem}\n\n"
        f"# Solution to the problem statement:\n{solution}\n\n"
        "Score this submission across Novelty, Usefulness, Feasibility, Scalability, and Sustainability. "
        "If attachments are present, summarize what they show in Image_contains_summary. "
        "Respond in the required JSON format."
    )


# =====================================================================
#  MAIN CODE LOGIC
# =====================================================================


def _build_system_instructions() -> str:
    schema = {
        "task": {
            "role": get_system_role(),
            "objective": get_evaluation_objective(),
            "method": get_evaluation_method(),
        },
        "schema": {
            "name": "Evaluation Schema",
            "parameters": get_scoring_criteria(),
        },
        "output_rules": {
            "format": "Return ONLY valid JSON object",
            "required_keys": METRICS,
            "structure": {
                m: {"score": "1-10", "reason": "string (one line why this score)"}
                for m in METRICS
            },
        },
    }

    examples = {"name": "Example Dataset", "list": get_few_shot_examples()}

    return (
        "# Evaluation System Schema\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "# Few-Shot Examples\n"
        f"{json.dumps(examples, indent=2, ensure_ascii=False)}\n\n"
        "# You must output a single JSON object with scores for each metric (score + reason). "
        "If images are provided, also include an Image_contains_summary field."
    )


def _to_direct_url(url: str) -> str:
    url = url.strip()
    file_id = None

    m = re.match(r"https://drive\.google\.com/file/d/([^/]+)", url)
    if m:
        file_id = m.group(1)

    if not file_id:
        m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
        if m and ("drive.google.com" in url or "drive.usercontent.google.com" in url):
            file_id = m.group(1)

    if file_id:
        return f"https://lh3.googleusercontent.com/d/{file_id}"

    return url


def _generate_id(length: int = 5) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _write_excel(results: List[dict], output_path: str, use_images: bool) -> None:
    """
    With images: CID | Problem | Solution | Image URL | Novelty | Reason | ... | Image Summary
    Without images: CID | Problem | Solution | Novelty | Reason | ...
    """
    from openpyxl import Workbook

    criteria = ["Novelty", "Usefulness", "Feasibility", "Scalability", "Sustainability"]

    wb = Workbook()
    ws = wb.active

    # Header row
    header = ["CID", "Problem", "Solution"]
    if use_images:
        header.append("Image URL")
    for c in criteria:
        header.extend([c, "Reason"])
    if use_images:
        header.append("Image Summary")
    ws.append(header)

    for entry in results:
        row = [
            str(entry.get("CID", "")),
            entry.get("Problem", ""),
            entry.get("Solution", ""),
        ]

        if use_images:
            image_url = entry.get("Image URL", "")
            if isinstance(image_url, list):
                image_url = ", ".join(image_url)
            row.append(image_url)

        scores = entry.get("scores", {})
        for c in criteria:
            obj = scores.get(c, {})
            if isinstance(obj, dict):
                row.append(obj.get("score", ""))
                row.append(obj.get("reason", ""))
            else:
                row.extend(["", ""])

        if use_images:
            row.append(scores.get("Image_contains_summary", ""))

        ws.append(row)

    wb.save(output_path)
    print(f"Excel file saved: {output_path}")


def run(
    provider: str,
    model: str,
    input_file: str,
    output_path: str,
    use_images: bool,
    temperature: float,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Provider: {provider}")
    print(f"  Model:    {model}")
    print(f"  Images:   {'YES' if use_images else 'NO'}")
    print(f"  Input:    {input_file}")
    print(f"  Output:   {output_path}")
    print(f"{'='*60}\n")

    input_path = Path(os.path.join(BASE_DIR, input_file))
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    ext = input_path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    else:
        print(f"ERROR: Unsupported file type '{ext}'. Use .csv, .xlsx, or .xls")
        return

    df = df.dropna(how="all")
    df.columns = df.columns.str.lower().str.strip()

    if "problem" not in df.columns or "solution" not in df.columns:
        print("ERROR: Input file must have 'problem' and 'solution' columns")
        return

    instructions = _build_system_instructions()

    output_schema = {
        "type": "object",
        "properties": {
            m: {
                "type": "object",
                "properties": {
                    "score": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["score", "reason"],
                "additionalProperties": False,
            }
            for m in METRICS
        },
        "required": METRICS + (["Image_contains_summary"] if use_images else []),
        "additionalProperties": False,
    }
    if use_images:
        output_schema["properties"]["Image_contains_summary"] = {"type": "string"}

    completion_config = KaapiCompletionConfig(
        provider=provider,
        type="text",
        params=TextLLMParams(
            model=model,
            instructions=instructions,
            temperature=temperature,
            output_schema=output_schema,
        ).model_dump(exclude_none=True),
    )

    completion_config, warnings = transform_kaapi_config_to_native(completion_config)
    configuration = LLMCallConfig(blob=ConfigBlob(completion=completion_config))
    provider_class = LLMProvider.get_provider_class(provider_type=provider)

    if provider == "openai":
        credential = {"api_key": os.getenv("OPENAI_API_KEY")}
    else:
        credential = {"api_key": os.getenv("GEMINI_API_KEY")}

    client = provider_class.create_client(credentials=credential)
    provider_instance = provider_class(client=client)

    results = []
    total = len(df)

    for idx, row in df.iterrows():
        if pd.isna(row.get("problem")) or pd.isna(row.get("solution")):
            continue

        cid = str(row["cid"]) if pd.notna(row.get("cid")) else f"student_{_generate_id(6)}"
        problem = str(row["problem"])
        solution = str(row["solution"])

        image_urls = None
        if pd.notna(row.get("documents")):
            image_urls = [u.strip() for u in str(row["documents"]).split(",") if u.strip()]
            image_urls = [_to_direct_url(u) for u in image_urls]

        if not image_urls:
            continue

        print(f"  [{idx+1}/{total}] Processing CID: {cid}...", end=" ", flush=True)

        user_text = get_user_prompt(problem, solution)
        inputs = [{"type": "text", "content": {"format": "text", "value": user_text}}]

        if use_images and image_urls:
            for img_url in image_urls:
                inputs.append({"type": "image", "content": {"format": "url", "value": img_url}})

        query = QueryParams(input=inputs)
        request = LLMCallRequest(query=query, config=configuration)

        try:
            with resolved_input_context(query_input=request.query.input) as resolved_input:
                response, error = provider_instance.execute(
                    completion_config=request.config.blob.completion,
                    query=query,
                    resolved_input=resolved_input,
                    include_provider_raw_response=False,
                )

            if error is not None or response is None:
                print(f"ERROR: {error}")
                response_output = None
            else:
                response_output = json.loads(response.response.output.content.value)
                print("OK")
        except Exception as e:
            print(f"EXCEPTION: {e}")
            response_output = None

        result = {
            "CID": cid,
            "Problem": problem,
            "Solution": solution,
            "scores": response_output if response_output else {},
        }

        if use_images:
            result["Image URL"] = image_urls if image_urls else []

        results.append(result)

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save JSON backup
    json_path = output_path.replace(".xlsx", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"JSON backup saved: {json_path}")

    # Save Excel
    _write_excel(results, output_path, use_images)

    print(f"\nDone! Processed {len(results)} submissions.")


if __name__ == "__main__":

    PROVIDER = "openai"
    MODEL = "gpt-4o-mini"
    INPUT_FILE = "200 Golden_dataset_2.O-3.xlsx"
    USE_IMAGES = False
    TEMPERATURE = 0.4

    suffix = "with_images" if USE_IMAGES else "wo_images"
    OUTPUT_PATH = os.path.join(BASE_DIR, "output", f"score_{suffix}.xlsx")

    run(
        provider=PROVIDER,
        model=MODEL,
        input_file=INPUT_FILE,
        output_path=OUTPUT_PATH,
        use_images=USE_IMAGES,
        temperature=TEMPERATURE,
    )

    USE_IMAGES = True
    TEMPERATURE = 0.4

    suffix = "with_images" if USE_IMAGES else "wo_images"
    OUTPUT_PATH = os.path.join(BASE_DIR, "output", f"score_{suffix}.xlsx")

    run(
        provider=PROVIDER,
        model=MODEL,
        input_file=INPUT_FILE,
        output_path=OUTPUT_PATH,
        use_images=USE_IMAGES,
        temperature=TEMPERATURE,
    )

    
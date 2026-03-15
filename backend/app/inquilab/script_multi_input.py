"""
Scoring rubric and message builder for the School Innovation Marathon (SIM).
Evaluates student innovations across 5 dimensions with Indian educational context.
"""

from typing import Dict, Any, List
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

load_dotenv()


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
                        You evaluate with the perspective of someone who has seen thousands of student innovations and can distinguish between genuine innovation, good effort, and superficial attempts.

                        CRITICAL SCORING GUARDRAILS:
                        - Scores MUST be justified by explicit evidence in the submission.
                        - Do NOT assume resources, infrastructure, or execution capability unless stated.
                        - If the idea and problem donot relate to each other then score least to all parameters.
                        - Generic textbook or widely known ideas must be scored low on novelty unless clear original adaptation is shown.

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
- Read the problem statement carefully - understand what issue the student is trying to solve
- Read the solution description - understand their proposed approach
- Identify the student's level of detail, originality, and thinking depth

STEP 2: CONTEXTUAL ASSESSMENT
- Consider this is a school student's work (grades 6-12)
- Assess against the backdrop of Indian educational and social context
- Look for evidence of personal observation vs. generic problem identification
- Evaluate solution practicality within Indian resource constraints

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
                "problem": "In case two trains accident in the railway station and another trains to go to another place .they got relay in India Odisha tripal train accident to solve this problem train accident prevention is is good Idea.",
                "solution": "Matter. The train automatically detects if another train is coming from the using infrared waves and stop the engine when ever it detect some object in front automatic train collisions prevention system a ground breaking solution design to and hands railway safety by preventing solution between trains equipped with infrared sensors this innovative system to detects approaching trains and automatically halts the engine upon detecting any obstruction ahead by utilising sensors to EMIT signal and detect the reflection from object in front the system effectively identities incoming trains and triggers the Indian shutdown mechanisms through a relay with both trains equipped with these sensor collisions are awarded as each train stop upon detecting the othd in its path\n Benefit. Simplified operation requires minimal manual intervention providing a user friendly solution for railway safety\n Experience the next level of railway safety with the automatic train collisions prevention on system .",
                "scores": {
                    "Novelty": {
                        "score": 5.0,
                        "reason": "Infrared collision detection is a known concept, but the student frames it as a system integration idea, which justifies mid-level originality.",
                    },
                    "Usefulness": {
                        "score": 6.0,
                        "reason": "The solution clearly targets a real safety problem and describes a functional prevention mechanism, even if simplified.",
                    },
                    "Feasibility": {
                        "score": 6.0,
                        "reason": "Sensor-based detection is technically achievable, but railway-grade deployment complexity limits higher scoring.",
                    },
                    "Scalability": {
                        "score": 7.0,
                        "reason": "The concept could be applied across rail systems if standardized, supporting moderate scaling potential.",
                    },
                    "Sustainability": {
                        "score": 6.0,
                        "reason": "Sensor-based automation reduces accident risk long-term, but maintenance and integration demands prevent higher scoring.",
                    },
                },
            },
            {
                "problem": "The who suffering from Asthma or breathing issues in crowded areas, they get immediate inhaler by wrist watch if they forgot to carry his/her personal inhaler.",
                "solution": "Immediate wrist watch inhaler for the person who needs it's instantly in the crowded areas when he or she will forgot to carry personal inhaler.",
                "scores": {
                    "Novelty": {
                        "score": 8.0,
                        "reason": "A wearable inhaler concept shows strong creative framing beyond common assistive devices.",
                    },
                    "Usefulness": {
                        "score": 7.0,
                        "reason": "The idea directly addresses emergency breathing scenarios with clear user benefit.",
                    },
                    "Feasibility": {
                        "score": 9.0,
                        "reason": "Miniaturized wearable medical tech is realistic with current engineering capabilities.",
                    },
                    "Scalability": {
                        "score": 8.0,
                        "reason": "The product could serve a wide patient population with minimal redesign.",
                    },
                    "Sustainability": {
                        "score": 9.0,
                        "reason": "Wearable medical devices are maintainable and long-lasting with proper servicing.",
                    },
                },
            },
            {
                "problem": "Manual painting of high-rise buildings is a hazardous, time-consuming, and costly process. Develop an Autonomous Robotic Painting System (ARPS) to improve safety, efficiency, and quality while reducing environmental impact and labor costs",
                "solution": "# Autonomous Robotic Painting System (ARPS)\n Key Features:\n 1. Robotic Arm: A multi-jointed arm with a painting tool attachment.\n 2. Navigation System: Sensors, GPS, and software for precise movement and painting control.\n 3. Painting Tank: A pressurized tank that stores and supplies paint to the robotic arm.\n 4. Safety Features: Emergency shutdown, stabilizers, and protective barriers.\n Benefits:\n 1. Improved Safety: Reduced risk of accidents and injuries.\n 2. Increased Efficiency: Faster painting times and reduced labor costs.\n 3. Enhanced Quality: Consistent, high-quality paint application.\n 4. Reduced Environmental Impact: Minimized paint waste and reduced VOC emissions.",
                "scores": {
                    "Novelty": {
                        "score": 2.0,
                        "reason": "Robotic painting is a widely known industrial automation concept with little student innovation.",
                    },
                    "Usefulness": {
                        "score": 3.0,
                        "reason": "Benefits are described, but mostly generic efficiency claims without unique implementation.",
                    },
                    "Feasibility": {
                        "score": 6.0,
                        "reason": "Industrial robotic systems are achievable but complex, justifying mid feasibility.",
                    },
                    "Scalability": {
                        "score": 3.0,
                        "reason": "High cost and infrastructure requirements limit broad adoption.",
                    },
                    "Sustainability": {
                        "score": 3.0,
                        "reason": "Environmental benefit claims are minimal and not operationally detailed.",
                    },
                },
            },
            {
                "problem": "Railway accidents caused by obstacles on tracks pose a significant threat to safety. This project utilizes an Arduino Uno, ultrasonic sensor, and buzzer to detect obstacles within a predefined range and alert train operators in real-time, preventing collisions and ensuring safety.",
                "solution": "The proposed solution involves implementing an obstacle detection system using Arduino Uno, an ultrasonic sensor, and a buzzer. The ultrasonic sensor continuously monitors the railway tracks for obstacles by emitting ultrasonic waves and measuring the time it takes for the waves to return. If an obstacle is detected within a predefined distance, the Arduino Uno processes the data and triggers the buzzer to alert the train operator. This system provides real-time warnings, enabling prompt action to avoid collisions.\n The solution is cost-effective, easy to deploy, and can function in diverse environmental conditions, making it suitable for both urban and remote railway networks. By integrating this system into railway operations, accidents caused by undetected obstacles can be significantly reduced, ensuring passenger and crew safety while minimizing infrastructure damage and service disruptions.",
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
                "problem": "Distance Calculator Project\n Problem Statement:\n Imagine you're planning a road trip. You want to know the distance between your starting point and your destination. You could use a map, but wouldn't it be much easier to have a simple tool that calculates the distance for you?\n Project Idea: Distance Calculator\n Description:\n This project involves creating a simple distance calculator that can determine the distance between two points on a 2D plane. This can be a great way to introduce students to basic programming concepts and mathematical formulas.\n Key Concepts:\n  * Distance Formula: The core of the project. This formula calculates the distance between t\n\n  * User Input: Students can learn to take input from the user, such as the coordinates of the two points.\n  * Mathematical Operations: The project involves basic mathematical operations like subtraction, squaring, and square root.\n  *",
                "solution": "Distance Calculator Project\n Problem Statement:\n Imagine you're planning a road trip. You want to know the distance between your starting point and your destination. You could use a map, but wouldn't it be much easier to have a simple tool that calculates the distance for you?\n Project Idea: Distance Calculator\n Description:\n This project involves creating a simple distance calculator that can determine the distance between two points on a 2D plane. This can be a great way to introduce students to basic programming concepts and mathematical formulas.\n Key Concepts:\n  * Distance Formula: The core of the project. This formula calculates the distance between t\n\n  * User Input: Students can learn to take input from the user, such as the coordinates of the two points.\n  * Mathematical Operations: The project involves basic mathematical operations like subtraction, squaring, and square root.\n  *",
                "scores": {
                    "Novelty": {
                        "score": 1.0,
                        "reason": "A distance calculator is a textbook programming exercise with no innovation.",
                    },
                    "Usefulness": {
                        "score": 2.0,
                        "reason": "The tool solves a minor convenience problem with limited impact.",
                    },
                    "Feasibility": {
                        "score": 2.0,
                        "reason": "While easy to implement, the rubric penalizes trivial execution lacking real-world relevance.",
                    },
                    "Scalability": {
                        "score": 2.0,
                        "reason": "The concept offers little expansion beyond its initial use case.",
                    },
                    "Sustainability": {
                        "score": 5.0,
                        "reason": "Software tools can persist over time, but value is modest.",
                    },
                },
            },
            {
                "problem": "Coal mining is a hazardous occupation that exposes workers to dangerous conditions, such as toxic gases, high temperatures, and poor visibility. Accidents caused by gas leaks, overheating, or other unsafe conditions can result in injuries or fatalities.",
                "solution": "the Smart Helmet for Coal Miners is designed as a wearable device equipped with sensors to monitor environmental conditions in real-time. The system will measure gas concentrations, temperature, and provide immediate alerts to ensure the safety of miners.",
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


def system_instruction_image() -> str:
    return """
IMAGE CONSIDERATION:

An image has been provided alongside the problem and solution text.

STEP 1: VALIDATE
- Check if the image is related to the student's project (prototype, model, circuit, diagram, poster, working demo, etc.)
- If the image is NOT project-related (selfie, random photo, meme, blank, unrelated screenshot),
  return ONLY this and nothing else:
  {"image_audit": "This image cannot be considered for evaluation"}
  Do NOT include any scores. Stop here.

STEP 2: USE AS SCORING CONTEXT
If the image IS project-related, factor it into your scoring:
- A visible working prototype or demo strengthens Feasibility evidence
- Original creative work shown in image strengthens Novelty evidence
- A well-built model or detailed diagram strengthens Usefulness evidence
- Evidence of real-world testing or deployment strengthens Scalability evidence
- Use of sustainable materials or eco-friendly approach visible in image strengthens Sustainability evidence
- If the image contradicts the text description, weigh that negatively in relevant scores

The image is supplementary evidence — use it to inform your scores within the same evaluation schema.
"""


def inference(
    *,
    problem: str,
    solution: str,
    image: str | list[str] | None = None,
    file: str | list[str] | None = None,
) -> str:
    solution_schema = get_evaluation_system_schema()
    examples_dataset = get_evaluation_examples()
    image_instruction = system_instruction_image()

    instructions = (
        "# Evaluation System Schema\n"
        f"{json.dumps(solution_schema, indent=2)}\n\n"
        "# Image Evaluation Guidelines\n"
        f"{image_instruction}\n\n"
        "# Few-Shot Examples\n"
        f"{json.dumps(examples_dataset, indent=2)}\n\n"
        "# Output should be strictly in json format"
    )

    completion_config = KaapiCompletionConfig(
        provider="openai",
        type="text",
        params=TextLLMParams(
            model="gpt-5.4", instructions=instructions, response_format="json_object"
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


def run_inference_batch(input: Path | List[dict[str, str]]) -> None:
    REQUIRED_COLUMNS = {"problem", "solution"}
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "output_multi/inference_batch")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    OUTPUT_FILE = "parsed_resultv1.json"
    OUTPUT_FILE_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)

    def _read_and_validate_file(file_path: Path) -> pd.DataFrame:
        ext = file_path.suffix.lower()
        file_data = None
        if ext == ".csv":
            file_data = pd.read_csv(file_path)
        elif ext in {".xlsx", ".xls"}:
            file_data = pd.read_excel(file_path)

        else:
            raise ValueError(f"Unsupported file type '{ext}'. Use .csv, .xlsx, or .xls")

        file_data.columns = file_data.columns.str.lower().str.strip()

        missing = REQUIRED_COLUMNS - set(file_data.keys())
        if missing:
            raise ValueError(f"Input dict must have both 'problem' and 'solution' keys")

        return file_data

    def _validate_dict_input(input_data: List[dict[str, str]]) -> None:
        for item in input_data:
            missing = REQUIRED_COLUMNS - set(item.keys())
            if missing:
                raise ValueError(
                    "Input dict must have both 'problem' and 'solution' keys"
                )
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

    data: List[dict[str, str]] | None = None

    if isinstance(input, Path):
        if os.path.exists(input):
            data = _read_and_validate_file(input)

            data = [
                EvaluationInput(
                    problem=row["problem"],
                    solution=row["solution"],
                    custom_id=str(row["cid"]) if pd.notna(row.get("cid")) else None,
                    image=str(row["image"]) if pd.notna(row.get("image")) else None,
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
        image_instruction = system_instruction_image()

        instructions = (
            "# Evaluation System Schema\n"
            f"{json.dumps(solution_schema, indent=2)}\n\n"
            "# Image Evaluation Guidelines\n"
            f"{image_instruction}\n\n"
            "# Few-Shot Examples\n"
            f"{json.dumps(examples_dataset, indent=2)}\n\n"
            "# Output should be strictly in json format"
        )

        jsonl_data = []

        for input_dict in data:
            input_dict = input_dict.model_dump()
            custom_id = (
                input_dict.get("custom_id") or f"student_{_generate_id(length=6)}"
            )
            problem = input_dict.get("problem")
            solution = input_dict.get("solution")
            image = input_dict.get("image", None)

            inputs = [
                {
                    "type": "input_text",
                    "text": f"# Problem Statement\n:{problem}\n\n# Solution to the problem statement:\n{solution}",
                }
            ]

            if image is not None:
                images = [image] if isinstance(image, str) else image
                for img in images:
                    inputs.append({"type": "input_image", "image_url": img})

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": "gpt-4o-mini",
                    "instructions": instructions,
                    "input": f"# Problem Statement\n:{problem}\n\n# Solution to the problem statement:\n{solution}",
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "score",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "novelty_score": {"type": "string"},
                                    "usefulness_score": {"type": "string"},
                                    "feasibility_score": {"type": "string"},
                                    "scalability_score": {"type": "string"},
                                    "sustainability_score": {"type": "string"},
                                },
                                "required": [
                                    "novelty_score",
                                    "usefulness_score",
                                    "feasibility_score",
                                    "scalability_score",
                                    "sustainability_score",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                },
            }

            jsonl_data.append(request)

        # Submit batch to openai
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        provider = OpenAIBatchProvider(client=openai_client)
        # create batch job
        result = provider.create_batch(jsonl_data, {})

        # wait for batch job to get suceed
        batch_id = result.get("provider_batch_id", "")
        batch_data: dict | None = None
        if batch_id:
            while True:
                batch_result = provider.get_batch_status(batch_id)
                if batch_result.get("provider_status") in (
                    "completed",
                    "failed",
                    "expired",
                    "cancelled",
                ):
                    batch_data = batch_result
                    break

            output_file_id = batch_data.get("provider_output_file_id", None)
            if not output_file_id:
                raise ValueError("No output file — batch may have failed")

            content = provider.download_batch_results(output_file_id)

            parsed_results = []
            for item in content:
                cid = item["custom_id"]
                if item.get("error"):
                    parsed_results.append({"custom_id": cid, "error": item["error"]})
                else:
                    text = item["response"]["body"]["output"][0]["content"][0]["text"]
                    parsed_results.append(
                        {"custom_id": cid, "scores": json.loads(text)}
                    )

            with open(OUTPUT_FILE_PATH, "w") as f:
                json.dump(parsed_results, f, indent=2)

        else:
            raise ValueError("batch id didn't got")


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
        "------- CASE 1: PROBLEM SOLUTION TEXT INPUT  ---------\n"
        + str(response_case1)
        + "\n"
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
        "------- CASE 3: MULTI-LINGUAL PROBLEM SOLUTION STATEMENT  ---------\n"
        + str(response_case3)
        + "\n"
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
        "------- CASE 4: MULTI-LINGUAL PROBLEM SOLUTION STATEMENT WITH IMAGE  ---------\n"
        + str(response_case4)
    )

    output_file = os.path.join(OUTPUT_FOLDER, "test_cases_output.txt")
    with open(output_file, "w") as f:
        f.write("".join(outputs))


if __name__ == "__main__":
    # TODO: uncomment to run inference test cases
    # run_inference_test_cases()

    run_inference_batch(Path(os.path.join(BASE_DIR, "mutli_input_batch.xlsx")))

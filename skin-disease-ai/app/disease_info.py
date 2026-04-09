"""
Disease Information Module
==========================
Comprehensive metadata for all 7 HAM10000 skin disease classes.
Includes descriptions, symptoms, causes, recommended actions, and risk levels.
"""

# Disease class labels (matching HAM10000 dataset encoding)
CLASS_NAMES = [
    'akiec',  # Actinic Keratoses / Intraepithelial Carcinoma
    'bcc',    # Basal Cell Carcinoma
    'bkl',    # Benign Keratosis-like Lesions
    'df',     # Dermatofibroma
    'mel',    # Melanoma
    'nv',     # Melanocytic Nevi
    'vasc'    # Vascular Lesions
]

# Human-readable names
CLASS_LABELS = {
    'akiec': 'Actinic Keratoses',
    'bcc':   'Basal Cell Carcinoma',
    'bkl':   'Benign Keratosis',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevi',
    'vasc':  'Vascular Lesions'
}

# Risk levels for each disease class
RISK_LEVELS = {
    'akiec': 'Medium',
    'bcc':   'High',
    'bkl':   'Low',
    'df':    'Low',
    'mel':   'High',
    'nv':    'Low',
    'vasc':  'Medium'
}

# Comprehensive disease information
DISEASE_INFO = {
    'akiec': {
        'name': 'Actinic Keratoses (Solar Keratoses)',
        'risk_level': 'Medium',
        'risk_color': '#f59e0b',
        'description': (
            'Actinic keratoses are rough, scaly patches on the skin caused by years of '
            'sun exposure. They are considered pre-cancerous lesions that can potentially '
            'progress to squamous cell carcinoma if left untreated. They most commonly '
            'appear on sun-exposed areas such as the face, ears, neck, scalp, chest, '
            'backs of hands, and forearms.'
        ),
        'symptoms': [
            'Rough, dry, or scaly patch of skin, usually less than 1 inch (2.5 cm) in diameter',
            'Flat to slightly raised patch or bump on the top layer of skin',
            'Color variations including pink, red, or brown',
            'Itching, burning, or tenderness in the affected area',
            'Hard, wart-like surface texture',
            'New patches may appear over time in sun-exposed areas'
        ],
        'causes': [
            'Chronic exposure to ultraviolet (UV) radiation from the sun',
            'Frequent use of tanning beds',
            'Fair skin, light hair, and light-colored eyes increase risk',
            'Weakened immune system',
            'Age over 40 with history of sun exposure',
            'Living in sunny climates or at high altitudes'
        ],
        'recommended_actions': [
            'Consult a dermatologist for professional evaluation',
            'May require cryotherapy (freezing), topical medications, or photodynamic therapy',
            'Regular skin examinations to monitor for changes',
            'Use broad-spectrum sunscreen (SPF 30+) daily',
            'Wear protective clothing and seek shade during peak sun hours',
            'Do not attempt to self-treat or remove the lesion'
        ]
    },

    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'risk_level': 'High',
        'risk_color': '#ef4444',
        'description': (
            'Basal cell carcinoma (BCC) is the most common form of skin cancer, arising '
            'from the basal cells in the lowest layer of the epidermis. While it rarely '
            'metastasizes, it can cause significant local tissue destruction if not treated '
            'promptly. BCC typically develops on areas frequently exposed to the sun, '
            'particularly the head and neck.'
        ),
        'symptoms': [
            'Pearly or waxy bump, often with visible blood vessels',
            'Flat, flesh-colored or brown scar-like lesion',
            'Bleeding or oozing sore that heals and then reopens',
            'Lesion with a slightly raised, translucent border',
            'Pink growth with a slightly elevated rolled border and crusted center',
            'Open sore that does not heal within a few weeks'
        ],
        'causes': [
            'Long-term exposure to ultraviolet (UV) radiation',
            'History of severe sunburns, especially in childhood',
            'Fair complexion with light eyes and hair',
            'Exposure to arsenic or radiation therapy',
            'Genetic conditions such as basal cell nevus syndrome',
            'Immunosuppression from medications or disease'
        ],
        'recommended_actions': [
            '⚠️ SEEK IMMEDIATE MEDICAL ATTENTION from a dermatologist',
            'Biopsy may be required for definitive diagnosis',
            'Treatment options include surgical excision, Mohs surgery, or radiation',
            'Regular follow-up examinations (every 6-12 months)',
            'Strict sun protection measures going forward',
            'Monitor for new suspicious lesions elsewhere on the body'
        ]
    },

    'bkl': {
        'name': 'Benign Keratosis (Seborrheic Keratosis)',
        'risk_level': 'Low',
        'risk_color': '#22c55e',
        'description': (
            'Benign keratosis-like lesions include seborrheic keratoses, solar lentigines, '
            'and lichen planus-like keratoses. Seborrheic keratoses are very common, '
            'noncancerous skin growths that appear as waxy, raised, tan to dark brown '
            'patches. They are harmless and do not require treatment unless they become '
            'irritated or are cosmetically concerning.'
        ),
        'symptoms': [
            'Waxy, stuck-on appearance, like a drop of candle wax on the skin',
            'Round or oval-shaped growths',
            'Color ranges from light tan to dark brown or black',
            'Slightly raised with a flat or rough texture',
            'Varies in size from very small to more than 1 inch across',
            'May itch but is typically painless'
        ],
        'causes': [
            'Exact cause is not fully understood',
            'Tend to develop with age (more common after age 50)',
            'Genetic predisposition may play a role',
            'Not caused by sun exposure, though may appear on sun-exposed areas',
            'Hormonal changes may trigger growth',
            'Skin friction in folded areas'
        ],
        'recommended_actions': [
            'Generally no treatment is necessary',
            'Consult a dermatologist if the lesion changes rapidly or bleeds',
            'Can be removed for cosmetic reasons via cryotherapy or curettage',
            'Monitor for any significant changes in shape, color, or size',
            'Differentiation from melanoma may require professional examination',
            'Take photos to track any changes over time'
        ]
    },

    'df': {
        'name': 'Dermatofibroma',
        'risk_level': 'Low',
        'risk_color': '#22c55e',
        'description': (
            'Dermatofibroma is a common, benign fibrous skin nodule that most often '
            'appears on the legs. These firm, round bumps are composed of fibrous tissue '
            'and are typically harmless. They may occur after minor injuries such as '
            'insect bites or small puncture wounds. Dermatofibromas rarely require '
            'treatment unless they cause discomfort.'
        ),
        'symptoms': [
            'Small, firm, raised bump on the skin',
            'Usually less than 1 cm in diameter',
            'Color ranges from pink to light brown to dark brown',
            'May dimple inward when pinched (dimple sign)',
            'Typically painless but may be tender to the touch',
            'Surface may feel smooth or slightly rough'
        ],
        'causes': [
            'May develop after minor skin injuries (insect bites, splinters)',
            'Overgrowth of fibrous tissue in the dermis',
            'More common in women than men',
            'Exact trigger is often unknown',
            'May be related to immune system response',
            'Not related to sun exposure'
        ],
        'recommended_actions': [
            'Usually no treatment is needed',
            'Consult a dermatologist for confirmation of diagnosis',
            'Surgical excision can be performed if bothersome',
            'Monitor for significant changes in size or color',
            'Cryotherapy may flatten the lesion if desired',
            'Avoid irritating the area with tight clothing'
        ]
    },

    'mel': {
        'name': 'Melanoma',
        'risk_level': 'High',
        'risk_color': '#ef4444',
        'description': (
            'Melanoma is the most serious and potentially life-threatening form of skin '
            'cancer. It develops in melanocytes, the cells that produce melanin (skin '
            'pigment). Melanoma can occur anywhere on the body, even in areas not exposed '
            'to the sun. Early detection is critical, as melanoma can spread rapidly to '
            'other organs if not treated in its early stages.'
        ),
        'symptoms': [
            'A new mole or change in an existing mole',
            'Asymmetry — one half of the mole does not match the other',
            'Border irregularity — edges are ragged, notched, or blurred',
            'Color variation — uneven shades of brown, black, red, white, or blue',
            'Diameter greater than 6mm (size of a pencil eraser)',
            'Evolving — the mole is changing in size, shape, or color over time'
        ],
        'causes': [
            'Exposure to ultraviolet (UV) radiation from sun or tanning beds',
            'History of severe, blistering sunburns',
            'Having many moles (more than 50) or atypical moles',
            'Fair skin, freckling, and light hair',
            'Family history of melanoma',
            'Weakened immune system'
        ],
        'recommended_actions': [
            '🚨 SEEK URGENT MEDICAL ATTENTION — See a dermatologist IMMEDIATELY',
            'Do NOT delay — early detection dramatically improves survival rates',
            'A biopsy is essential for definitive diagnosis',
            'Treatment may include surgery, immunotherapy, or targeted therapy',
            'Regular full-body skin examinations every 3-6 months',
            'Strict sun protection is critical going forward'
        ]
    },

    'nv': {
        'name': 'Melanocytic Nevi (Moles)',
        'risk_level': 'Low',
        'risk_color': '#22c55e',
        'description': (
            'Melanocytic nevi, commonly known as moles, are benign growths on the skin '
            'formed by clusters of melanocytes. They are extremely common — most adults '
            'have between 10 to 40 moles. Moles can be flat or raised, and their color '
            'ranges from pink to dark brown. While most moles are harmless, it is '
            'important to monitor them for changes that could indicate melanoma.'
        ),
        'symptoms': [
            'Small, round or oval spots on the skin',
            'Uniform color (tan, brown, black, pink, or skin-toned)',
            'Distinct border separating the mole from surrounding skin',
            'Usually less than 6mm in diameter',
            'Can be flat or raised, smooth or rough',
            'May develop hairs growing from within'
        ],
        'causes': [
            'Genetic predisposition to developing moles',
            'Sun exposure, especially during childhood',
            'Fair skin is associated with more moles',
            'Hormonal changes (puberty, pregnancy) can affect moles',
            'Clusters of melanocytes forming during development',
            'Most develop during childhood and adolescence'
        ],
        'recommended_actions': [
            'Regular self-examination using the ABCDE rule',
            'Consult a dermatologist if any changes occur',
            'Monitor moles for asymmetry, border changes, color variation',
            'Annual professional skin examination recommended',
            'Protect skin with sunscreen and clothing',
            'Photograph moles to track changes over time'
        ]
    },

    'vasc': {
        'name': 'Vascular Lesions',
        'risk_level': 'Medium',
        'risk_color': '#f59e0b',
        'description': (
            'Vascular lesions are abnormalities of blood vessels in the skin. This '
            'category includes cherry angiomas, angiokeratomas, and pyogenic granulomas. '
            'Most vascular lesions are benign, but some may require evaluation to rule '
            'out more serious conditions. They can appear as red, purple, or blue spots '
            'or bumps on the skin.'
        ),
        'symptoms': [
            'Red, purple, or blue patches or nodules on the skin',
            'May bleed easily if scratched or injured',
            'Can be flat or raised, ranging from pinpoint to several centimeters',
            'Cherry angiomas appear as small, bright red domes',
            'Some lesions may throb or ache',
            'Color may change with pressure (blanching)'
        ],
        'causes': [
            'Abnormal proliferation of blood vessels',
            'Genetic factors and hereditary conditions',
            'Aging — cherry angiomas increase with age',
            'Hormonal changes during pregnancy',
            'Trauma or injury to the skin',
            'Some types may be associated with liver disease'
        ],
        'recommended_actions': [
            'Consult a dermatologist for proper diagnosis',
            'Most vascular lesions are benign and do not require treatment',
            'Laser therapy or electrocautery can remove lesions if desired',
            'Seek medical attention if lesions bleed frequently',
            'Monitor for rapid growth or changes in appearance',
            'Rule out other vascular conditions with professional evaluation'
        ]
    }
}


def get_disease_info(class_code):
    """
    Get comprehensive information about a disease by its class code.

    Args:
        class_code (str): The disease class code (e.g., 'mel', 'nv', 'bcc')

    Returns:
        dict: Disease information including name, risk level, description,
              symptoms, causes, and recommended actions.
    """
    if class_code in DISEASE_INFO:
        return DISEASE_INFO[class_code]
    return None


def get_risk_level(class_code):
    """Get the risk level for a disease class."""
    return RISK_LEVELS.get(class_code, 'Unknown')


def get_class_label(class_code):
    """Get the human-readable label for a class code."""
    return CLASS_LABELS.get(class_code, 'Unknown')

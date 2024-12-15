list_m2m = [
    # Germanic: 
    # Afrikaans, Danish, Dutch, German, English, Icelandic, Luxembourgish, Norwegian, Swedish, Frisian, Yiddish
    "af", "da", "nl", "de", "en", "is", "lb", "no", "sv", "fy", "yi",
    # Romance: 
    # Asturian, Catalan, French, Galician, Italian, Occitan, Portuguese, Romanian, Spanish
    "ast", "ca", "fr", "gl", "it", "oc", "pt", "ro", "es",
    # Slavic: 
    # Belarusian, Bosnian, Bulgarian, Croatian, Czech, Macedonian, Polish, Russian, Serbian, Slovak, Slovenian, Ukrainian
    "be", "bs", "bg", "hr", "cs", "mk", "pl", "ru", "sr", "sk", "sl", "uk",
    # Uralic and Baltic: 
    # Estonian, Finnish, Hungarian, Latvian, Lithuanian
    "et", "fi", "hu", "lv", "lt",
    # Albanian: Albanian, Armenian: Armenian, Kartvelian: Georgian, Hellenic: Greek
    "sq", "hy", "ka", "el",
    # Celtic:
    # Breton Irish(family: Irish), Scottish Gaelic, Welsh
    "br", "ga", "gd", "cy",
    # Turkic:
    # Azerbaijani, Bashkir, Kazakh, Turkish, Uzbek
    "az", "ba", "kk", "tr", "uz",
    # Japonic: Japanese, Koreanic: Korean, Vietic: Vietnamese, Chinese: Chinese
    "ja", "ko", "vi", "zh",
    # Indo-Aryan:
    # Bengali, Gujarati, Hindi, Marathi, 
    # Nepali, Oriya, Panjabi, Sindhi, Sinhala, Urdu
    "bn", "gu", "hi", "mr", "ne", "or", "pa", "sd", "si", "ur",
    # Malayo-Polynesian:
    # Cebuano, Ilocano, Indonesian, Javanese, Malagasy, Malay, Sundanese, Tagalog
    "ceb", "ilo", "id", "jv", "mg", "ms", "su", "tl",
    # Sino-Tibetan: Burmese, Khmer: Khmer, Kra-Dai: Lao, Thai, Mongolic: Mongolian
    "my", "km", "lo", "th", "mn",
    # Arabic: Arabic, Semitic: Hebrew, Iranian: Pashto, Persian, 
    "ar", "he", "ps", "fa",
    # Tamil and Dravidian: Kannada, Tamil, Malayalam
    "kn", "ta", "ml",
    # Ethiopian: Amharic
    # Niger-Congo: Fulfulde, Hausa, Igbo, Lingala, Luganda, Northern Sotho, Somali(Cushitic), Swahili, Swati, Tswana, Wolof, Xhosa, Yoruba, Zulu
    "am", "ff", "ha", "ig", "ln", "lg", "ns", "so", "sw", "ss", "tn", "wo", "xh", "yo", "zu",
    # Creole: Haitian Creole
    "ht"
]

m2m_dict = {
    # Germanic
    "Afrikaans": "af", "Danish": "da", "Dutch": "nl", "German": "de", "English": "en", "Icelandic": "is", "Luxembourgish": "lb", 
    "Norwegian": "no", "Swedish": "sv", "Frisian": "fy", "Yiddish": "yi",
    # Romance
    "Asturian": "ast", "Catalan": "ca", "French": "fr", "Galician": "gl", "Italian": "it", "Occitan": "oc", "Portuguese": "pt", 
    "Romanian": "ro", "Spanish": "es",
    # Slavic
    "Belarusian": "be", "Bosnian": "bs", "Bulgarian": "bg", "Croatian": "hr", "Czech": "cs", "Macedonian": "mk", "Polish": "pl", 
    "Russian": "ru", "Serbian": "sr", "Slovak": "sk", "Slovenian": "sl", "Ukrainian": "uk",
    # Uralic and Baltic
    "Estonian": "et", "Finnish": "fi", "Hungarian": "hu", "Latvian": "lv", "Lithuanian": "lt",
    # Albanian, Armenian, Kartvelian, Hellenic
    "Albanian": "sq", "Armenian": "hy", "Georgian": "ka", "Greek": "el",
    # Celtic
    "Breton":"br", "Irish": "ga", "Scottish Gaelic": "gd", "Welsh": "cy",
    # Turkic
    "Azerbaijani": "az", "Bashkir": "ba", "Kazakh": "kk", "Turkish": "tr", "Uzbek": "uz",
    # Japonic, Koreanic, Vietic, Chinese
    "Japanese": "ja", "Korean": "ko", "Vietnamese": "vi", "Chinese": "zh",
    # Indo-Aryan
    "Bengali": "bn", "Gujarati": "gu", "Hindi": "hi", "Kannada": "kn", "Marathi": "mr", "Nepali": "ne", "Oriya": "or", "Panjabi": "pa", 
    "Sindhi": "sd", "Sinhala": "si", "Urdu": "ur", "Tamil": "ta",
    # Malayo-Polynesian
    "Cebuano": "ceb", "Ilocano": "ilo", "Indonesian": "id", "Javanese": "jv", "Malagasy": "mg", "Malay": "ms", "Malayalam": "ml", "Sundanese": "su", "Tagalog": "tl",
    # Sino-Tibetan, Khmer, Kra-Dai, Thai, Mongolic
    "Burmese": "my", "Khmer": "km", "Lao": "lo", "Thai":"th", "Mongolian": "mn",
    # Arabic, Semitic, Iranian
    "Arabic": "ar", "Hebrew": "he", "Pashto": "ps", "Persian": "fa",
    # Ethiopian
    "Amharic": "am",
    # Niger-Congo
    "Fulfulde": "ff", "Hausa": "ha", "Igbo": "ig", "Lingala": "ln", "Luganda": "lg", "Northern Sotho": "ns", "Somali": "so", 
    "Swahili": "sw", "Swati": "ss", "Tswana": "tn", "Wolof": "wo", "Xhosa": "xh", "Yoruba": "yo", "Zulu": "zu",
    # Creole
    "Haitian Creole": "ht"
}
nllb_dict = {
    "Thai": "tha_Thai",
    "Acehnese (Arabic script)": "ace_Arab",
    "Acehnese (Latin script)": "ace_Latn",
    "Mesopotamian Arabic": "acm_Arab",
    "Ta’izzi-Adeni Arabic": "acq_Arab",
    "Tunisian Arabic": "aeb_Arab",
    "Afrikaans": "afr_Latn",
    "South Levantine Arabic": "ajp_Arab",
    "Akan": "aka_Latn",
    "Amharic": "amh_Ethi",
    "North Levantine Arabic": "apc_Arab",
    "Arabic": "arb_Arab",
    "Modern Standard Arabic (Romanized)": "arb_Latn",
    "Najdi Arabic": "ars_Arab",
    "Moroccan Arabic": "ary_Arab",
    "Egyptian Arabic": "arz_Arab",
    "Assamese": "asm_Beng",
    "Asturian": "ast_Latn",
    "Awadhi": "awa_Deva",
    "Central Aymara": "ayr_Latn",
    "Azerbaijani": "azj_Latn",
    "Bashkir": "bak_Cyrl",
    "Bambara": "bam_Latn",
    "Balinese": "ban_Latn",
    "Belarusian": "bel_Cyrl",
    "Bemba": "bem_Latn",
    "Bengali": "ben_Beng",
    "Bhojpuri": "bho_Deva",
    "Banjar (Arabic script)": "bjn_Arab",
    "Banjar (Latin script)": "bjn_Latn",
    "Standard Tibetan": "bod_Tibt",
    "Bosnian": "bos_Latn",
    "Buginese": "bug_Latn",
    "Bulgarian": "bul_Cyrl",
    "Catalan": "cat_Latn",
    "Cebuano": "ceb_Latn",
    "Czech": "ces_Latn",
    "Chokwe": "cjk_Latn",
    "Central Kurdish": "ckb_Arab",
    "Crimean Tatar": "crh_Latn",
    "Welsh": "cym_Latn",
    "Danish": "dan_Latn",
    "German": "deu_Latn",
    "Southwestern Dinka": "dik_Latn",
    "Dyula": "dyu_Latn",
    "Dzongkha": "dzo_Tibt",
    "Greek": "ell_Grek",
    "English": "eng_Latn",
    "Esperanto": "epo_Latn",
    "Estonian": "est_Latn",
    "Basque": "eus_Latn",
    "Ewe": "ewe_Latn",
    "Faroese": "fao_Latn",
    "Fijian": "fij_Latn",
    "Finnish": "fin_Latn",
    "Fon": "fon_Latn",
    "French": "fra_Latn",
    "Friulian": "fur_Latn",
    "Fulfulde": "fuv_Latn",
    "Scottish Gaelic": "gla_Latn",
    "Irish": "gle_Latn",
    "Galician": "glg_Latn",
    "Guarani": "grn_Latn",
    "Gujarati": "guj_Gujr",
    "Haitian Creole": "hat_Latn",
    "Hausa": "hau_Latn",
    "Hebrew": "heb_Hebr",
    "Hindi": "hin_Deva",
    "Chhattisgarhi": "hne_Deva",
    "Croatian": "hrv_Latn",
    "Hungarian": "hun_Latn",
    "Armenian": "hye_Armn",
    "Igbo": "ibo_Latn",
    "Ilocano": "ilo_Latn",
    "Indonesian": "ind_Latn",
    "Icelandic": "isl_Latn",
    "Italian": "ita_Latn",
    "Javanese": "jav_Latn",
    "Japanese": "jpn_Jpan",
    "Kabyle": "kab_Latn",
    "Jingpho": "kac_Latn",
    "Kamba": "kam_Latn",
    "Kannada": "kan_Knda",
    "Kashmiri (Arabic script)": "kas_Arab",
    "Kashmiri (Devanagari script)": "kas_Deva",
    "Georgian": "kat_Geor",
    "Central Kanuri (Arabic script)": "knc_Arab",
    "Central Kanuri (Latin script)": "knc_Latn",
    "Kazakh": "kaz_Cyrl",
    "Kabiyè": "kbp_Latn",
    "Kabuverdianu": "kea_Latn",
    "Khmer": "khm_Khmr",
    "Kikuyu": "kik_Latn",
    "Kinyarwanda": "kin_Latn",
    "Kyrgyz": "kir_Cyrl",
    "Kimbundu": "kmb_Latn",
    "Northern Kurdish": "kmr_Latn",
    "Kikongo": "kon_Latn",
    "Korean": "kor_Hang",
    "Lao": "lao_Laoo",
    "Ligurian": "lij_Latn",
    "Limburgish": "lim_Latn",
    "Lingala": "lin_Latn",
    "Lithuanian": "lit_Latn",
    "Lombard": "lmo_Latn",
    "Latgalian": "ltg_Latn",
    "Luxembourgish": "ltz_Latn",
    "Luba-Kasai": "lua_Latn",
    "Luganda": "lug_Latn",
    "Luo": "luo_Latn",
    "Mizo": "lus_Latn",
    "Latvian": "lvs_Latn",
    "Magahi": "mag_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Marathi": "mar_Deva",
    "Minangkabau (Arabic script)": "min_Arab",
    "Minangkabau (Latin script)": "min_Latn",
    "Macedonian": "mkd_Cyrl",
    "Malagasy": "plt_Latn",
    "Maltese": "mlt_Latn",
    "Meitei (Bengali script)": "mni_Beng",
    "Mongolian": "khk_Cyrl",
    "Mossi": "mos_Latn",
    "Maori": "mri_Latn",
    "Burmese": "mya_Mymr",
    "Dutch": "nld_Latn",
    "Norwegian": "nob_Latn",
    "Nepali": "npi_Deva",
    "Northern Sotho": "nso_Latn",
    "Nuer": "nus_Latn",
    "Nyanja": "nya_Latn",
    "Occitan": "oci_Latn",
    "West Central Oromo": "gaz_Latn",
    "Oriya": "ory_Orya",
    "Pangasinan": "pag_Latn",
    "Panjabi": "pan_Guru",
    "Papiamento": "pap_Latn",
    "Persian": "pes_Arab",
    "Polish": "pol_Latn",
    "Portuguese": "por_Latn",
    "Dari": "prs_Arab",
    "Pashto": "pbt_Arab",
    "Ayacucho Quechua": "quy_Latn",
    "Romanian": "ron_Latn",
    "Rundi": "run_Latn",
    "Russian": "rus_Cyrl",
    "Sango": "sag_Latn",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sicilian": "scn_Latn",
    "Shan": "shn_Mymr",
    "Sinhala": "sin_Sinh",
    "Slovak": "slk_Latn",
    "Slovenian": "slv_Latn",
    "Samoan": "smo_Latn",
    "Shona": "sna_Latn",
    "Sindhi": "snd_Arab",
    "Somali": "som_Latn",
    "Southern Sotho": "sot_Latn",
    "Spanish": "spa_Latn",
    "Albanian": "als_Latn",
    "Sardinian": "srd_Latn",
    "Serbian": "srp_Cyrl",
    "Swati": "ssw_Latn",
    "Sundanese": "sun_Latn",
    "Swedish": "swe_Latn",
    "Swahili": "swh_Latn",
    "Silesian": "szl_Latn",
    "Tamil": "tam_Taml",
    "Tatar": "tat_Cyrl",
    "Telugu": "tel_Telu",
    "Tajik": "tgk_Cyrl",
    "Tagalog": "tgl_Latn",
    "Tigrinya": "tir_Ethi",
    "Tamasheq (Latin script)": "taq_Latn",
    "Tamasheq (Tifinagh script)": "taq_Tfng",
    "Tok Pisin": "tpi_Latn",
    "Tswana": "tsn_Latn",
    "Tsonga": "tso_Latn",
    "Turkmen": "tuk_Latn",
    "Tumbuka": "tum_Latn",
    "Turkish": "tur_Latn",
    "Twi": "twi_Latn",
    "Central Atlas Tamazight": "tzm_Tfng",
    "Uyghur": "uig_Arab",
    "Ukrainian": "ukr_Cyrl",
    "Umbundu": "umb_Latn",
    "Urdu": "urd_Arab",
    "Uzbek": "uzn_Latn",
    "Venetian": "vec_Latn",
    "Vietnamese": "vie_Latn",
    "Waray": "war_Latn",
    "Wolof": "wol_Latn",
    "Xhosa": "xho_Latn",
    "Yiddish": "ydd_Hebr",
    "Yoruba": "yor_Latn",
    "Chinese": "zho_Hans",
    "Malay": "zsm_Latn",
    "Zulu": "zul_Latn"
}




def translate_language_code(input_str, source_type, output_type):
    if source_type not in ["m2m", "nllb", "language"] or output_type not in ["m2m", "nllb", "language", "flores"]:
        raise ValueError("Invalid source_type or output_type")
    
    if source_type == "language":
        language_name = input_str
    elif source_type == "m2m":
        language_name = next((name for name, code in m2m_dict.items() if code == input_str), None)
    elif source_type == "nllb":
        language_name = next((name for name, code in nllb_dict.items() if code == input_str), None)
    else:
        raise ValueError("Invalid source_type")

    if language_name is None:
        return None

    if output_type == "language":
        return language_name
    elif output_type == "m2m":
        return m2m_dict.get(language_name, None)
    elif output_type == "nllb":
        return nllb_dict.get(language_name, None)
    elif output_type == "flores":
        code = nllb_dict.get(language_name, None)
        if code == "zho_Hans": 
            return "cmn_Hans"
        elif code == "est_Latn":
            return "ekk_Latn"
        elif code == "tgl_Latn":
            return "fil_Latn"
        else:
            return code
    else:
        raise ValueError("Invalid output_type")
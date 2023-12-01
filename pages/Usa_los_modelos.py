# import spacy_streamlit
import base64
from itertools import islice
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy
import pandas as pd
import spacy
import streamlit as st
import typer
from packaging.version import Version
from spacy import displacy
from spacy import Language
from spacy.language import Language
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.scorer import PRFScore
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.tokens.doc import Doc
from spacy.training.example import Example
from spacy.vocab import Vocab
from thinc.api import chain
from thinc.api import Linear
from thinc.api import Logistic
from thinc.api import Model
from thinc.api import Optimizer
from thinc.model import set_dropout_rate
from thinc.types import cast
from thinc.types import Floats2d
from thinc.types import Ints1d
from thinc.types import Ragged
from wasabi import Printer

# from .util import load_model, process_text, get_svg, get_html, LOGO





# make the factory work
# from rel_pipe import make_relation_extractor, score_relations

# make the config work
# from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors

SPACY_VERSION = Version(spacy.__version__)

# fmt: off
NER_ATTRS = ["text", "label_", "start", "end", "start_char", "end_char"]
TOKEN_ATTRS = ["idx", "text", "lemma_", "pos_", "tag_", "dep_", "head", "morph",
               "ent_type_", "ent_iob_", "shape_", "is_alpha", "is_ascii",
               "is_digit", "is_punct", "like_num", "is_sent_start"]
# Currently these attrs are the same, but they might differ in the future.
SPAN_ATTRS = NER_ATTRS

# fmt: on
FOOTER = """<span style="font-size: 0.75em">&hearts; Tec de Monterrey 2023</span><br><span style="font-size: 0.65em">Luis José González</span><br><span style="font-size: 0.65em">Fernando Arana</span><br><span style="font-size: 0.65em">Sofía Hernández</span><br><span style="font-size: 0.75em">Abiel Borja</span><br><span style="font-size: 0.65em">Julieta Noguez</span><br><span style="font-size: 0.65em">Patricia Caratozzolo</span>"""


def visualize(
    models: Union[List[str], Dict[str, str]],
    default_text: str = "",
    default_model: Optional[str] = None,
    visualizers: List[str] = ["parser", "ner", "textcat", "similarity", "tokens"],
    ner_labels: Optional[List[str]] = None,
    ner_attrs: List[str] = NER_ATTRS,
    similarity_texts: Tuple[str, str] = ("apple", "orange"),
    token_attrs: List[str] = TOKEN_ATTRS,
    show_json_doc: bool = True,
    show_meta: bool = True,
    show_config: bool = True,
    show_visualizer_select: bool = False,
    show_pipeline_info: bool = True,
    sidebar_title: Optional[str] = None,
    sidebar_description: Optional[str] = None,
    show_logo: bool = True,
    color: Optional[str] = "#09A3D5",
    key: Optional[str] = None,
    get_default_text: Callable[[Language], str] = None,
) -> None:
    """Embed the full visualizer with selected components.

    :param models: Union[List[str]:
    :param Dict[str:
    :param str]]:
    :param default_text: str:  (Default value = "")
    :param default_model: Optional[str]:  (Default value = None)
    :param visualizers: List[str]:  (Default value = ["parser")
    :param "ner":
    :param "textcat":
    :param "similarity":
    :param "tokens"]:
    :param ner_labels: Optional[List[str]]:  (Default value = None)
    :param ner_attrs: List[str]:  (Default value = NER_ATTRS)
    :param similarity_texts: Tuple[str:
    :param str]:  (Default value = None)
    :param "orange"):
    :param token_attrs: List[str]:  (Default value = TOKEN_ATTRS)
    :param show_json_doc: bool:  (Default value = True)
    :param show_meta: bool:  (Default value = True)
    :param show_config: bool:  (Default value = True)
    :param show_visualizer_select: bool:  (Default value = False)
    :param show_pipeline_info: bool:  (Default value = True)
    :param sidebar_title: Optional[str]:  (Default value = None)
    :param sidebar_description: Optional[str]:  (Default value = None)
    :param show_logo: bool:  (Default value = True)
    :param color: Optional[str]:  (Default value = "#09A3D5")
    :param key: Optional[str]:  (Default value = None)
    :param get_default_text: Callable[[Language]:

    """

    if st.config.get_option("theme.primaryColor") != color:
        st.config.set_option("theme.primaryColor", color)

        # Necessary to apply theming
        st.experimental_rerun()

    if show_logo:
        st.sidebar.markdown(LOGO, unsafe_allow_html=True)
    if sidebar_title:
        st.sidebar.title(sidebar_title)
    if sidebar_description:
        st.sidebar.markdown(sidebar_description)

    # Allow both dict of model name / description as well as list of names
    model_names = models
    format_func = str
    if isinstance(models, dict):

        def format_func(name):
            """

            :param name:

            """
            return models.get(name, name)

        model_names = list(models.keys())

    default_model_index = (
        model_names.index(default_model)
        if default_model is not None and default_model in model_names
        else 0
    )
    spacy_model = st.sidebar.selectbox(
        "Model",
        model_names,
        index=default_model_index,
        key=f"{key}_visualize_models",
        format_func=format_func,
    )
    model_load_state = st.info(f"Loading model '{spacy_model}'...")
    nlp = load_model(spacy_model)
    model_load_state.empty()

    if show_pipeline_info:
        st.sidebar.subheader("Pipeline info")
        desc = f"""<p style="font-size: 0.85em; line-height: 1.5"><strong>{spacy_model}:</strong> <code>v{nlp.meta['version']}</code>. {nlp.meta.get("description", "")}</p>"""
        st.sidebar.markdown(desc, unsafe_allow_html=True)

    if show_visualizer_select:
        active_visualizers = st.sidebar.multiselect(
            "Visualizers",
            options=visualizers,
            default=list(visualizers),
            key=f"{key}_viz_select",
        )
    else:
        active_visualizers = visualizers

    default_text = (
        get_default_text(nlp) if get_default_text is not None else default_text
    )
    text = st.text_area(
        "Ingresa el texto para analizar:", default_text, key=f"{key}_visualize_text"
    )
    doc = process_text(spacy_model, text)

    if "parser" in visualizers and "parser" in active_visualizers:
        visualize_parser(doc, key=key)
    if "ner" in visualizers and "ner" in active_visualizers:
        ner_labels = ner_labels or nlp.get_pipe("ner").labels
        visualize_ner(
            doc,
            labels=ner_labels,
            attrs=ner_attrs,
            key=key,
            colors={
                "SKILL": "#d1aaff",
                "OCC": "linear-gradient(90deg, #fff176, #ffee58)",
            },
        )
    if "rel" in visualizers and "rel" in active_visualizers:
        ner_labels = ner_labels or nlp.get_pipe("ner").labels
        visualize_rel(doc)
    if "textcat" in visualizers and "textcat" in active_visualizers:
        visualize_textcat(doc)
    if "similarity" in visualizers and "similarity" in active_visualizers:
        visualize_similarity(nlp, default_texts=similarity_texts, key=key)
    if "tokens" in visualizers and "tokens" in active_visualizers:
        visualize_tokens(doc, attrs=token_attrs, key=key)

    if show_json_doc or show_meta or show_config:
        st.header("Pipeline information")
        if show_json_doc:
            json_doc_exp = st.expander("JSON Doc")
            json_doc_exp.json(doc.to_json())

        if show_meta:
            meta_exp = st.expander("Pipeline meta.json")
            meta_exp.json(nlp.meta)

        if show_config:
            config_exp = st.expander("Pipeline config.cfg")
            config_exp.code(nlp.config.to_str())

    st.sidebar.markdown(
        FOOTER,
        unsafe_allow_html=True,
    )


def visualize_parser(
    doc: Union[spacy.tokens.Doc, List[Dict[str, str]]],
    *,
    title: Optional[str] = "Dependency Parse & Part-of-speech tags",
    key: Optional[str] = None,
    manual: bool = False,
    displacy_options: Optional[Dict] = None,
) -> None:
    """Visualizer for dependency parses.

    doc (Doc, List): The document to visualize.
    key (str): Key used for the streamlit component for selecting labels.
    title (str): The title displayed at the top of the parser visualization.
    manual (bool): Flag signifying whether the doc argument is a Doc object or a List of Dicts containing parse information.
    displacy_options (Dict): Dictionary of options to be passed to the displacy render method for generating the HTML to be rendered.
      See: https://spacy.io/api/top-level#options-dep

    :param doc: Union[spacy.tokens.Doc:
    :param List[Dict[str:
    :param str]]]:
    :param *:
    :param title: Optional[str]:  (Default value = "Dependency Parse & Part-of-speech tags")
    :param key: Optional[str]:  (Default value = None)
    :param manual: bool:  (Default value = False)
    :param displacy_options: Optional[Dict]:  (Default value = None)

    """
    if displacy_options is None:
        displacy_options = dict()
    if title:
        st.header(title)
    if manual:
        # In manual mode, collapse_phrases and collapse_punct are passed as options to
        # displacy.parse_deps(doc) and the resulting data is retokenized to be correct,
        # so we already have these options configured at the time we use this data.
        cols = st.columns(1)
        split_sents = False
        options = {
            "compact": cols[0].checkbox("Compact mode", key=f"{key}_parser_compact"),
        }
    else:
        cols = st.columns(4)
        split_sents = cols[0].checkbox(
            "Split sentences", value=True, key=f"{key}_parser_split_sents"
        )
        options = {
            "collapse_punct": cols[1].checkbox(
                "Collapse punct", value=True, key=f"{key}_parser_collapse_punct"
            ),
            "collapse_phrases": cols[2].checkbox(
                "Collapse phrases", key=f"{key}_parser_collapse_phrases"
            ),
            "compact": cols[3].checkbox("Compact mode", key=f"{key}_parser_compact"),
        }
    docs = [span.as_doc() for span in doc.sents] if split_sents else [doc]
    # add selected options to options provided by user
    # `options` from `displacy_options` are overwritten by user provided
    # options from the checkboxes
    displacy_options = {**displacy_options, **options}
    for sent in docs:
        html = displacy.render(
            sent, options=displacy_options, style="dep", manual=manual
        )
        # Double newlines seem to mess with the rendering
        html = html.replace("\n\n", "\n")
        if split_sents and len(docs) > 1:
            st.markdown(f"> {sent.text}")
        st.write(get_svg(html), unsafe_allow_html=True)


def visualize_rel(doc: Union[spacy.tokens.Doc, List[Dict[str, str]]]):
    """

    :param doc: Union[spacy.tokens.Doc:
    :param List[Dict[str:
    :param str]]]:

    """
    return 0


def visualize_ner(
    doc: Union[spacy.tokens.Doc, List[Dict[str, str]]],
    *,
    labels: Sequence[str] = tuple(),
    attrs: List[str] = NER_ATTRS,
    show_table: bool = True,
    title: Optional[str] = "Named Entities",
    colors: Dict[str, str] = {},
    key: Optional[str] = None,
    manual: bool = False,
    displacy_options: Optional[Dict] = None,
):
    """Visualizer for named entities.

    doc (Doc, List): The document to visualize.
    labels (list): The entity labels to visualize.
    attrs (list):  The attributes on the entity Span to be labeled. Attributes are displayed only when the show_table
    argument is True.
    show_table (bool): Flag signifying whether to show a table with accompanying entity attributes.
    title (str): The title displayed at the top of the NER visualization.
    colors (Dict): Dictionary of colors for the entity spans to visualize, with keys as labels and corresponding colors
    as the values. This argument will be deprecated soon. In future the colors arg need to be passed in the displacy_options arg
    with the key "colors".
    key (str): Key used for the streamlit component for selecting labels.
    manual (bool): Flag signifying whether the doc argument is a Doc object or a List of Dicts containing entity span
    information.
    displacy_options (Dict): Dictionary of options to be passed to the displacy render method for generating the HTML to be rendered.
      See https://spacy.io/api/top-level#displacy_options-ent.

    :param doc: Union[spacy.tokens.Doc:
    :param List[Dict[str:
    :param str]]]:
    :param *:
    :param labels: Sequence[str]:  (Default value = tuple())
    :param attrs: List[str]:  (Default value = NER_ATTRS)
    :param show_table: bool:  (Default value = True)
    :param title: Optional[str]:  (Default value = "Named Entities")
    :param colors: Dict[str:
    :param str]:  (Default value = {})
    :param key: Optional[str]:  (Default value = None)
    :param manual: bool:  (Default value = False)
    :param displacy_options: Optional[Dict]:  (Default value = None)

    """
    if not displacy_options:
        displacy_options = dict()
    if colors:
        displacy_options["colors"] = colors

    if title:
        st.subheader(title)

    if manual:
        if show_table:
            st.warning(
                "When the parameter 'manual' is set to True, the parameter 'show_table' must be set to False."
            )
        if not isinstance(doc, list):
            st.warning(
                "When the parameter 'manual' is set to True, the parameter 'doc' must be of type 'list', not 'spacy.tokens.Doc'."
            )
    else:
        labels = labels or list({ent.label_ for ent in doc.ents})

    if not labels:
        st.warning("The parameter 'labels' should not be empty or None.")
    else:
        exp = st.expander("Select entity labels")
        label_select = exp.multiselect(
            "Entity labels",
            options=labels,
            default=list(labels),
            key=f"{key}_ner_label_select",
        )

        displacy_options["ents"] = label_select
        html = displacy.render(
            doc,
            style="ent",
            options=displacy_options,
            manual=manual,
        )
        style = "<style>mark.entity { display: inline-block }</style>"
        st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)
        if show_table:
            data = [
                [str(getattr(ent, attr)) for attr in attrs]
                for ent in doc.ents
                if ent.label_ in label_select
            ]
            if data:
                df = pd.DataFrame(data, columns=attrs)
                st.dataframe(df)


def visualize_spans(
    doc: Union[spacy.tokens.Doc, Dict[str, str]],
    *,
    spans_key: str = "sc",
    attrs: List[str] = SPAN_ATTRS,
    show_table: bool = True,
    title: Optional[str] = "Spans",
    manual: bool = False,
    displacy_options: Optional[Dict] = None,
):
    """Visualizer for spans.

    doc (Doc, Dict): The document to visualize.
    spans_key (str): Which spans key to render spans from. Default is "sc".
    attrs (list):  The attributes on the entity Span to be labeled. Attributes are displayed only when the show_table
    argument is True.
    show_table (bool): Flag signifying whether to show a table with accompanying span attributes.
    title (str): The title displayed at the top of the Spans visualization.
    manual (bool): Flag signifying whether the doc argument is a Doc object or a List of Dicts containing span information.
    displacy_options (Dict): Dictionary of options to be passed to the displacy render method for generating the HTML to be rendered.
      See https://spacy.io/api/top-level#displacy_options-span

    :param doc: Union[spacy.tokens.Doc:
    :param Dict[str:
    :param str]]:
    :param *:
    :param spans_key: str:  (Default value = "sc")
    :param attrs: List[str]:  (Default value = SPAN_ATTRS)
    :param show_table: bool:  (Default value = True)
    :param title: Optional[str]:  (Default value = "Spans")
    :param manual: bool:  (Default value = False)
    :param displacy_options: Optional[Dict]:  (Default value = None)

    """
    if SPACY_VERSION < Version("3.3.0"):
        raise ValueError(
            f"'visualize_spans' requires spacy>=3.3.0. You have spacy=={spacy.__version__}"
        )
    if not displacy_options:
        displacy_options = dict()
    displacy_options["spans_key"] = spans_key

    if title:
        st.header(title)

    if manual:
        if show_table:
            st.warning(
                "When the parameter 'manual' is set to True, the parameter 'show_table' must be set to False."
            )
        if not isinstance(doc, dict):
            st.warning(
                "When the parameter 'manual' is set to True, the parameter 'doc' must be of type 'Dict', not 'spacy.tokens.Doc'."
            )
    html = displacy.render(
        doc,
        style="span",
        options=displacy_options,
        manual=manual,
    )
    st.write(f"{get_html(html)}", unsafe_allow_html=True)

    if show_table:
        data = [
            [str(getattr(span, attr)) for attr in attrs]
            for span in doc.spans[spans_key]
        ]
        if data:
            df = pd.DataFrame(data, columns=attrs)
            st.dataframe(df)


def visualize_textcat(
    doc: spacy.tokens.Doc, *, title: Optional[str] = "Text Classification"
) -> None:
    """Visualizer for text categories.

    :param doc: spacy.tokens.Doc:
    :param *:
    :param title: Optional[str]:  (Default value = "Text Classification")

    """
    if title:
        st.header(title)
    st.markdown(f"> {doc.text}")
    df = pd.DataFrame(doc.cats.items(), columns=("Label", "Score"))
    st.dataframe(df)


def visualize_similarity(
    nlp: spacy.language.Language,
    default_texts: Tuple[str, str] = ("apple", "orange"),
    *,
    threshold: float = 0.5,
    title: Optional[str] = "Vectors & Similarity",
    key: Optional[str] = None,
) -> None:
    """Visualizer for semantic similarity using word vectors.

    :param nlp: spacy.language.Language:
    :param default_texts: Tuple[str:
    :param str]:  (Default value = ("apple")
    :param "orange"):
    :param *:
    :param threshold: float:  (Default value = 0.5)
    :param title: Optional[str]:  (Default value = "Vectors & Similarity")
    :param key: Optional[str]:  (Default value = None)

    """
    meta = nlp.meta.get("vectors", {})
    if title:
        st.header(title)
    if not meta.get("width", 0):
        st.warning("No vectors available in the model.")
    else:
        cols = st.columns(2)
        text1 = cols[0].text_input(
            "Text or word 1", default_texts[0], key=f"{key}_similarity_text1"
        )
        text2 = cols[1].text_input(
            "Text or word 2", default_texts[1], key=f"{key}_similarity_text2"
        )
        doc1 = nlp.make_doc(text1)
        doc2 = nlp.make_doc(text2)
        similarity = doc1.similarity(doc2)
        similarity_text = f"**Score:** `{similarity}`"
        if similarity > threshold:
            st.success(similarity_text)
        else:
            st.error(similarity_text)

        exp = st.expander("Vector information")
        exp.code(meta)


def visualize_tokens(
    doc: spacy.tokens.Doc,
    *,
    attrs: List[str] = TOKEN_ATTRS,
    title: Optional[str] = "Token attributes",
    key: Optional[str] = None,
) -> None:
    """Visualizer for token attributes.

    :param doc: spacy.tokens.Doc:
    :param *:
    :param attrs: List[str]:  (Default value = TOKEN_ATTRS)
    :param title: Optional[str]:  (Default value = "Token attributes")
    :param key: Optional[str]:  (Default value = None)

    """
    if title:
        st.header(title)
    exp = st.expander("Select token attributes")
    selected = exp.multiselect(
        "Token attributes",
        options=attrs,
        default=list(attrs),
        key=f"{key}_tokens_attr_select",
    )
    data = [[str(getattr(token, attr)) for attr in selected] for token in doc]
    df = pd.DataFrame(data, columns=selected)
    st.dataframe(df)


@st.cache_resource
def load_model(name: str) -> spacy.language.Language:
    """Load a spaCy model.

    :param name: str:

    """
    return spacy.load(name)


@st.cache_data
def process_text(model_name: str, text: str) -> spacy.tokens.Doc:
    """Process a text and create a Doc object.

    :param model_name: str:
    :param text: str:

    """
    nlp = load_model(model_name)
    return nlp(text)


def get_svg(svg: str, style: str = "", wrap: bool = True):
    """Convert an SVG to a base64-encoded image.

    :param svg: str:
    :param style: str:  (Default value = "")
    :param wrap: bool:  (Default value = True)

    """
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    html = f'<img src="data:image/svg+xml;base64,{b64}" style="{style}"/>'
    return get_html(html) if wrap else html


def get_html(html: str):
    """Convert HTML so it can be rendered.

    :param html: str:

    """
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)


LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 900 500 175" width="150" height="53"><path fill="#09A3D5" d="M64.8 970.6c-11.3-1.3-12.2-16.5-26.7-15.2-7 0-13.6 2.9-13.6 9.4 0 9.7 15 10.6 24.1 13.1 15.4 4.7 30.4 7.9 30.4 24.7 0 21.3-16.7 28.7-38.7 28.7-18.4 0-37.1-6.5-37.1-23.5 0-4.7 4.5-8.4 8.9-8.4 5.5 0 7.5 2.3 9.4 6.2 4.3 7.5 9.1 11.6 21 11.6 7.5 0 15.3-2.9 15.3-9.4 0-9.3-9.5-11.3-19.3-13.6-17.4-4.9-32.3-7.4-34-26.7-1.8-32.9 66.7-34.1 70.6-5.3-.3 5.2-5.2 8.4-10.3 8.4zm81.5-28.8c24.1 0 37.7 20.1 37.7 44.9 0 24.9-13.2 44.9-37.7 44.9-13.6 0-22.1-5.8-28.2-14.7v32.9c0 9.9-3.2 14.7-10.4 14.7-8.8 0-10.4-5.6-10.4-14.7v-95.6c0-7.8 3.3-12.6 10.4-12.6 6.7 0 10.4 5.3 10.4 12.6v2.7c6.8-8.5 14.6-15.1 28.2-15.1zm-5.7 72.8c14.1 0 20.4-13 20.4-28.2 0-14.8-6.4-28.2-20.4-28.2-14.7 0-21.5 12.1-21.5 28.2.1 15.7 6.9 28.2 21.5 28.2zm59.8-49.3c0-17.3 19.9-23.5 39.2-23.5 27.1 0 38.2 7.9 38.2 34v25.2c0 6 3.7 17.9 3.7 21.5 0 5.5-5 8.9-10.4 8.9-6 0-10.4-7-13.6-12.1-8.8 7-18.1 12.1-32.4 12.1-15.8 0-28.2-9.3-28.2-24.7 0-13.6 9.7-21.4 21.5-24.1 0 .1 37.7-8.9 37.7-9 0-11.6-4.1-16.7-16.3-16.7-10.7 0-16.2 2.9-20.4 9.4-3.4 4.9-2.9 7.8-9.4 7.8-5.1 0-9.6-3.6-9.6-8.8zm32.2 51.9c16.5 0 23.5-8.7 23.5-26.1v-3.7c-4.4 1.5-22.4 6-27.3 6.7-5.2 1-10.4 4.9-10.4 11 .2 6.7 7.1 12.1 14.2 12.1zM354 909c23.3 0 48.6 13.9 48.6 36.1 0 5.7-4.3 10.4-9.9 10.4-7.6 0-8.7-4.1-12.1-9.9-5.6-10.3-12.2-17.2-26.7-17.2-22.3-.2-32.3 19-32.3 42.8 0 24 8.3 41.3 31.4 41.3 15.3 0 23.8-8.9 28.2-20.4 1.8-5.3 4.9-10.4 11.6-10.4 5.2 0 10.4 5.3 10.4 11 0 23.5-24 39.7-48.6 39.7-27 0-42.3-11.4-50.6-30.4-4.1-9.1-6.7-18.4-6.7-31.4-.4-36.4 20.8-61.6 56.7-61.6zm133.3 32.8c6 0 9.4 3.9 9.4 9.9 0 2.4-1.9 7.3-2.7 9.9l-28.7 75.4c-6.4 16.4-11.2 27.7-32.9 27.7-10.3 0-19.3-.9-19.3-9.9 0-5.2 3.9-7.8 9.4-7.8 1 0 2.7.5 3.7.5 1.6 0 2.7.5 3.7.5 10.9 0 12.4-11.2 16.3-18.9l-27.7-68.5c-1.6-3.7-2.7-6.2-2.7-8.4 0-6 4.7-10.4 11-10.4 7 0 9.8 5.5 11.6 11.6l18.3 54.3 18.3-50.2c2.7-7.8 3-15.7 12.3-15.7z" /> </svg>"""

LOGO = get_svg(LOGO_SVG, wrap=False, style="max-width: 100%; margin-bottom: 25px")


@spacy.registry.architectures("rel_model.v1")
def create_relation_model(
    create_instance_tensor: Model[List[Doc], Floats2d],
    classification_layer: Model[Floats2d, Floats2d],
) -> Model[List[Doc], Floats2d]:
    """

    :param create_instance_tensor: Model[List[Doc]:
    :param Floats2d]:
    :param classification_layer: Model[Floats2d:

    """
    with Model.define_operators({">>": chain}):
        model = create_instance_tensor >> classification_layer
        model.attrs["get_instances"] = create_instance_tensor.attrs["get_instances"]
    return model


@spacy.registry.architectures("rel_classification_layer.v1")
def create_classification_layer(
    nO: int = None, nI: int = None
) -> Model[Floats2d, Floats2d]:
    """

    :param nO: int:  (Default value = None)
    :param nI: int:  (Default value = None)

    """
    with Model.define_operators({">>": chain}):
        return Linear(nO=nO, nI=nI) >> Logistic()


@spacy.registry.misc("rel_instance_generator.v1")
def create_instances(max_length: int) -> Callable[[Doc], List[Tuple[Span, Span]]]:
    """

    :param max_length: int:

    """
    def get_instances(doc: Doc) -> List[Tuple[Span, Span]]:
        """

        :param doc: Doc:

        """
        instances = []
        for ent1 in doc.ents:
            for ent2 in doc.ents:
                if ent1 != ent2:
                    if max_length and abs(ent2.start - ent1.start) <= max_length:
                        instances.append((ent1, ent2))
        return instances

    return get_instances


# def get_instances(max_length: int, doc: Doc) -> List[Tuple[Span, Span]]:
#     instances = []
#     for ent1 in doc.ents:
#         for ent2 in doc.ents:
#             if ent1 != ent2:
#                 if max_length and abs(ent2.start - ent1.start) <= max_length:
#                     instances.append((ent1, ent2))
#     return instances

# @spacy.registry.misc("rel_instance_generator.v1")
# def create_instances(max_length: int) -> Callable[[Doc], List[Tuple[Span, Span]]]:
#     def instances_for_doc(doc: Doc) -> List[Tuple[Span, Span]]:
#         return get_instances(max_length, doc)

#     return instances_for_doc


@spacy.registry.architectures("rel_instance_tensor.v1")
def create_tensors(
    tok2vec: Model[List[Doc], List[Floats2d]],
    pooling: Model[Ragged, Floats2d],
    get_instances: Callable[[Doc], List[Tuple[Span, Span]]],
) -> Model[List[Doc], Floats2d]:
    """

    :param tok2vec: Model[List[Doc]:
    :param List[Floats2d]]:
    :param pooling: Model[Ragged:
    :param Floats2d]:
    :param get_instances: Callable[[Doc]:
    :param List[Tuple[Span:
    :param Span]]]:

    """
    return Model(
        "instance_tensors",
        instance_forward,
        layers=[tok2vec, pooling],
        refs={"tok2vec": tok2vec, "pooling": pooling},
        attrs={"get_instances": get_instances},
        init=instance_init,
    )


def instance_forward(
    model: Model[List[Doc], Floats2d], docs: List[Doc], is_train: bool
) -> Tuple[Floats2d, Callable]:
    """

    :param model: Model[List[Doc]:
    :param Floats2d]:
    :param docs: List[Doc]:
    :param is_train: bool:

    """
    pooling = model.get_ref("pooling")
    tok2vec = model.get_ref("tok2vec")
    get_instances = model.attrs["get_instances"]
    all_instances = [get_instances(doc) for doc in docs]
    tokvecs, bp_tokvecs = tok2vec(docs, is_train)

    ents = []
    lengths = []

    for doc_nr, (instances, tokvec) in enumerate(zip(all_instances, tokvecs)):
        token_indices = []
        for instance in instances:
            for ent in instance:
                token_indices.extend([i for i in range(ent.start, ent.end)])
                lengths.append(ent.end - ent.start)
        ents.append(tokvec[token_indices])
    lengths = cast(Ints1d, model.ops.asarray(lengths, dtype="int32"))
    entities = Ragged(model.ops.flatten(ents), lengths)
    pooled, bp_pooled = pooling(entities, is_train)

    # Reshape so that pairs of rows are concatenated
    relations = model.ops.reshape2f(pooled, -1, pooled.shape[1] * 2)

    def backprop(d_relations: Floats2d) -> List[Doc]:
        """

        :param d_relations: Floats2d:

        """
        d_pooled = model.ops.reshape2f(d_relations, d_relations.shape[0] * 2, -1)
        d_ents = bp_pooled(d_pooled).data
        d_tokvecs = []
        ent_index = 0
        for doc_nr, instances in enumerate(all_instances):
            shape = tokvecs[doc_nr].shape
            d_tokvec = model.ops.alloc2f(*shape)
            count_occ = model.ops.alloc2f(*shape)
            for instance in instances:
                for ent in instance:
                    d_tokvec[ent.start : ent.end] += d_ents[ent_index]
                    count_occ[ent.start : ent.end] += 1
                    ent_index += ent.end - ent.start
            d_tokvec /= count_occ + 0.00000000001
            d_tokvecs.append(d_tokvec)

        d_docs = bp_tokvecs(d_tokvecs)
        return d_docs

    return relations, backprop


def instance_init(model: Model, X: List[Doc] = None, Y: Floats2d = None) -> Model:
    """

    :param model: Model:
    :param X: List[Doc]:  (Default value = None)
    :param Y: Floats2d:  (Default value = None)

    """
    tok2vec = model.get_ref("tok2vec")
    if X is not None:
        tok2vec.initialize(X)
    return model


Doc.set_extension("rel", default={}, force=True)
msg = Printer()


@Language.factory(
    "relation_extractor",
    requires=["doc.ents", "token.ent_iob", "token.ent_type"],
    assigns=["doc._.rel"],
    default_score_weights={
        "rel_micro_p": None,
        "rel_micro_r": None,
        "rel_micro_f": None,
    },
)
def make_relation_extractor(
    nlp: Language, name: str, model: Model, *, threshold: float
):
    """Construct a RelationExtractor component.

    :param nlp: Language:
    :param name: str:
    :param model: Model:
    :param *:
    :param threshold: float:

    """
    return RelationExtractor(nlp.vocab, model, name, threshold=threshold)


class RelationExtractor(TrainablePipe):
    """ """
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "rel",
        *,
        threshold: float,
    ) -> None:
        """Initialize a relation extractor."""
        self.vocab = vocab
        self.model = model
        self.name = name
        self.cfg = {"labels": [], "threshold": threshold}

    @property
    def labels(self) -> Tuple[str]:
        """Returns the labels currently added to the component."""
        return tuple(self.cfg["labels"])

    @property
    def threshold(self) -> float:
        """Returns the threshold above which a prediction is seen as 'True'."""
        return self.cfg["threshold"]

    def add_label(self, label: str) -> int:
        """Add a new label to the pipe.

        :param label: str:

        """
        if not isinstance(label, str):
            raise ValueError(
                "Only strings can be added as labels to the RelationExtractor"
            )
        if label in self.labels:
            return 0
        self.cfg["labels"] = list(self.labels) + [label]
        return 1

    def __call__(self, doc: Doc) -> Doc:
        """Apply the pipe to a Doc."""
        # check that there are actually any candidate instances in this batch of examples
        total_instances = len(self.model.attrs["get_instances"](doc))
        if total_instances == 0:
            msg.info("Could not determine any instances in doc - returning doc as is.")
            return doc

        predictions = self.predict([doc])
        self.set_annotations([doc], predictions)
        return doc

    def predict(self, docs: Iterable[Doc]) -> Floats2d:
        """Apply the pipeline's model to a batch of docs, without modifying them.

        :param docs: Iterable[Doc]:

        """
        get_instances = self.model.attrs["get_instances"]
        total_instances = sum([len(get_instances(doc)) for doc in docs])
        if total_instances == 0:
            msg.info(
                "Could not determine any instances in any docs - can not make any predictions."
            )
        scores = self.model.predict(docs)
        return self.model.ops.asarray(scores)

    def set_annotations(self, docs: Iterable[Doc], scores: Floats2d) -> None:
        """Modify a batch of `Doc` objects, using pre-computed scores.

        :param docs: Iterable[Doc]:
        :param scores: Floats2d:

        """
        c = 0
        get_instances = self.model.attrs["get_instances"]
        for doc in docs:
            for e1, e2 in get_instances(doc):
                offset = (e1.start, e2.start)
                if offset not in doc._.rel:
                    doc._.rel[offset] = {}
                for j, label in enumerate(self.labels):
                    doc._.rel[offset][label] = scores[c, j]
                c += 1

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        set_annotations: bool = False,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Learn from a batch of documents and gold-standard information,
        updating the pipe's model. Delegates to predict and get_loss.

        :param examples: Iterable[Example]:
        :param *:
        :param drop: float:  (Default value = 0.0)
        :param set_annotations: bool:  (Default value = False)
        :param sgd: Optional[Optimizer]:  (Default value = None)
        :param losses: Optional[Dict[str:
        :param float]]:  (Default value = None)

        """
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        set_dropout_rate(self.model, drop)

        # check that there are actually any candidate instances in this batch of examples
        total_instances = 0
        for eg in examples:
            total_instances += len(self.model.attrs["get_instances"](eg.predicted))
        if total_instances == 0:
            msg.info("Could not determine any instances in doc.")
            return losses

        # run the model
        docs = [eg.predicted for eg in examples]
        predictions, backprop = self.model.begin_update(docs)
        loss, gradient = self.get_loss(examples, predictions)
        backprop(gradient)
        if sgd is not None:
            self.model.finish_update(sgd)
        losses[self.name] += loss
        if set_annotations:
            self.set_annotations(docs, predictions)
        return losses

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, float]:
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores.

        :param examples: Iterable[Example]:
        :param scores:

        """
        truths = self._examples_to_truth(examples)
        gradient = scores - truths
        mean_square_error = (gradient**2).sum(axis=1).mean()
        return float(mean_square_error), gradient

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Language = None,
        labels: Optional[List[str]] = None,
    ):
        """Initialize the pipe for training, using a representative set
        of data examples.

        :param get_examples: Callable[[]:
        :param Iterable[Example]]:
        :param *:
        :param nlp: Language:  (Default value = None)
        :param labels: Optional[List[str]]:  (Default value = None)

        """
        if labels is not None:
            for label in labels:
                self.add_label(label)
        else:
            for example in get_examples():
                relations = example.reference._.rel
                for indices, label_dict in relations.items():
                    for label in label_dict.keys():
                        self.add_label(label)
        self._require_labels()

        subbatch = list(islice(get_examples(), 10))
        doc_sample = [eg.reference for eg in subbatch]
        label_sample = self._examples_to_truth(subbatch)
        if label_sample is None:
            raise ValueError(
                "Call begin_training with relevant entities and relations annotated in "
                "at least a few reference examples!"
            )
        self.model.initialize(X=doc_sample, Y=label_sample)

    def _examples_to_truth(self, examples: List[Example]) -> Optional[numpy.ndarray]:
        """

        :param examples: List[Example]:

        """
        # check that there are actually any candidate instances in this batch of examples
        nr_instances = 0
        for eg in examples:
            nr_instances += len(self.model.attrs["get_instances"](eg.reference))
        if nr_instances == 0:
            return None

        truths = numpy.zeros((nr_instances, len(self.labels)), dtype="f")
        c = 0
        for i, eg in enumerate(examples):
            for e1, e2 in self.model.attrs["get_instances"](eg.reference):
                gold_label_dict = eg.reference._.rel.get((e1.start, e2.start), {})
                for j, label in enumerate(self.labels):
                    truths[c, j] = gold_label_dict.get(label, 0)
                c += 1

        truths = self.model.ops.asarray(truths)
        return truths

    def score(self, examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
        """Score a batch of examples.

        :param examples: Iterable[Example]:
        :param **kwargs:

        """
        return score_relations(examples, self.threshold)


# def score_relations(examples: Iterable[Example], threshold: float) -> Dict[str, Any]:
#     """Score a batch of examples."""
#     micro_prf = PRFScore()
#     for example in examples:
#         gold = example.reference._.rel
#         pred = example.predicted._.rel
#         for key, pred_dict in pred.items():
#             gold_labels = [k for (k, v) in gold.get(key, {}).items() if v == 1.0]
#             for k, v in pred_dict.items():
#                 if v >= threshold:
#                     if k in gold_labels:
#                         micro_prf.tp += 1
#                     else:
#                         micro_prf.fp += 1
#                 else:
#                     if k in gold_labels:
#                         micro_prf.fn += 1
#     return {
#         "rel_micro_p": micro_prf.precision,
#         "rel_micro_r": micro_prf.recall,
#         "rel_micro_f": micro_prf.fscore,
#     }


def score_relations(examples: Iterable[Example], threshold: float) -> Dict[str, Any]:
    """Score a batch of examples for NER.

    :param examples: Iterable[Example]:
    :param threshold: float:

    """
    micro_prf = PRFScore()
    for example in examples:
        gold = example.reference.ents
        pred = example.predicted.ents
        gold_labels = {ent.label_: (ent.start, ent.end) for ent in gold}
        pred_labels = {span.label_: (span.start, span.end) for span in pred}
        for label, pred_span in pred_labels.items():
            if label is not None:
                if label in gold_labels:
                    gold_span = gold_labels[label]
                    if pred_span == gold_span:
                        micro_prf.tp += 1
                    else:
                        micro_prf.fp += 1
                else:
                    micro_prf.fp += 1
        for label, gold_span in gold_labels.items():
            if label is not None and label not in pred_labels:
                micro_prf.fn += 1

    return {
        "ner_micro_p": micro_prf.precision,
        "ner_micro_r": micro_prf.recall,
        "ner_micro_f": micro_prf.fscore,
    }


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


st.set_page_config(
    page_title="Applied models",
    page_icon="💻",
)

text = "The Chief Information Security Officer develops and drives the vision for the information security function. He/She acts as the authority for the development and enforcement of organisation security strategy, standards and policies, and has ultimate responsibility for ensuring the protection of corporate information. He guides the design and continuous improvement of the IT security architecture and Cyber Risk Maturity Model that balances business needs with security risks. He advises the board and top executives on all security matters and sets directions for complying with regulatory inquiries, legal and compliance regulations, inspections and audits. He is an expert in cyber security compliance standards, protocols and frameworks, as well as the Cyber Security Act 2018. He is keeps abreast of cyber-related applications and hardware technologies and services, and is constantly on the look-out for new technologies that may be leveraged on to enhance work processes, or which may pose as potential threats. The Chief Information Security Officer is an inspirational and influential leader, who displays sound judgement and decisiveness in ensuring that corporate information is well protected and secured. He is strategic in his approach toward resource management and capability development among his teams."


def main(models: str, default_text: str):
    """

    :param models: str:
    :param default_text: str:

    """
    models = [name.strip() for name in models.split(",")]
    nlp = spacy.load(models[0])
    nlp.add_pipe("sentencizer")
    doc = nlp(text)
    visualizers = ["ner", "rel"]
    # spacy_streamlit.visualize_ner(doc,
    #                               colors={"SKILL": "#d1aaff", "OCC": "linear-gradient(90deg, #fff176, #ffee58)"})
    visualize(
        models,
        default_text,
        visualizers=visualizers,
        show_visualizer_select=True,
        show_logo=False,
        sidebar_title="Skill Taxonomies Model",
        sidebar_description="Elige uno de nuestros modelos para predecir Entidades y Relaciones.",
        color="#d1aaff",
    )

    st.title("Skill Taxonomies Model")


if __name__ == "__main__":
    main(models="./training/model-best,./training/model-last", default_text=text)
    # try:
    #     typer.run(main)
    # except SystemExit:
    #     pass

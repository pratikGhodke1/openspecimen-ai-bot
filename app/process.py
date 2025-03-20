import json
from dataclasses import dataclass
from copy import copy
import re

import numpy as np
import pandas as pd


@dataclass
class Page:
    markdown: str
    metadata: dict | None = None


@dataclass
class ExtractedPDF:
    file_name: str
    pages: list[Page]


@dataclass
class ParsedPDF:
    filename: str
    text: str
    pages_description: list[str]


def create_extracted_pages_object(extracted_pdfs, data):
    # extracted_page = ExtractedPage()

    for pdf in data:
        pdf = ParsedPDF(**pdf)
        parsed_pages = []

        for page in pdf.pages_description:
            parsed_pages.append(Page(page))

        if extracted_pdf := extracted_pdfs.get(pdf.filename):
            extracted_pdf.pages.extend(parsed_pages)
        else:
            extracted_pdfs[pdf.filename] = ExtractedPDF(
                file_name=pdf.filename, pages=copy(parsed_pages)
            )
    return extracted_pdfs


def extract_metadata(extracted_pdfs: dict[str, ExtractedPDF]) -> None:
    pattern = r'metadata\s*{\s*"subject":\s*"([^"]+)",\s*"chapter":\s*"([^"]*)",\s*"section":\s*\[\s*([^]]*)\s*\],\s*"page_number":\s*"(\d+)",\s*"page_description":\s*"([^"]+)"\s*}'

    for pdf_file_name, extracted_pdf_pages in sorted(extracted_pdfs.items()):
        print(f"{pdf_file_name}: {len(extracted_pdf_pages.pages)}")

        for i, page in enumerate(extracted_pdf_pages.pages):
            # print(f"\tpage={i}")
            page.markdown = re.sub("(\u2018|\u2019)", "'", page.markdown)
            page.markdown = re.sub("(\u2014)", "-", page.markdown)

            match = re.search(
                pattern,
                page.markdown,
                re.DOTALL,
            )

            # Extract and store in a dictionary if match is found
            if match:

                # Extract individual fields
                subject = match.group(1)
                chapter = match.group(2)
                raw_section = match.group(3).strip()
                page_number = match.group(4)
                page_description = match.group(5)

                # Parse the `section` field properly
                if raw_section:  # If section is not empty
                    try:
                        section = json.loads(
                            f"[{raw_section}]"
                        )  # Convert to Python list
                    except json.JSONDecodeError:
                        section = [raw_section]  # Fallback if parsing fails
                else:
                    section = []  # Empty list if no sections are present

                # Construct the metadata dictionary
                page.metadata = {
                    "subject": subject,
                    "chapter": chapter,
                    "section": section,
                    "page_number": page_number,
                    "page_description": page_description,
                }
                # print(page.metadata)
            else:
                page.metadata = {}
                # print(f"\tpage={i}")
                # print("\t No metadata found")
                # print("\t",page)

            # print(v.pages[0])
            # print("------------------")
    return extracted_pdfs


def is_nan(value):
    if not isinstance(value, float):
        return False
    return np.isnan(value)


current_header = []


def ffill_sections(x: list):
    global current_header
    if is_nan(x):
        x = current_header
    elif current_header and not len(x):
        x.extend(current_header)
    else:
        current_header = x

    return x


def update_metadata(extracted_pdf_pages: ExtractedPDF, processed_metadata):
    for pdf_page, meta in zip(extracted_pdf_pages.pages, processed_metadata):
        pdf_page.metadata = meta

        # Remove the existing metadata
        pattern = r'metadata\s*{\s*"subject":\s*"([^"]+)",\s*"chapter":\s*"([^"]*)",\s*"section":\s*\[\s*([^]]*)\s*\],\s*"page_number":\s*"(\d+)",\s*"page_description":\s*"([^"]+)"\s*}'
        # Append new metadata
        # print(json.dumps(meta))
        pdf_page.markdown = re.sub(
            pattern,
            json.dumps(meta),
            pdf_page.markdown,
        )


def  post_process_metadata(extracted_pdfs: dict[str, ExtractedPDF]) -> None:
    for pdf_file_name, extracted_pdf_pages in extracted_pdfs.items():
        print(f"Processing file: {pdf_file_name}")

        metadata_list = []

        for page in extracted_pdf_pages.pages:
            metadata_list.append(page.metadata)

        og = pd.DataFrame(metadata_list)
        try:
            # Fix page number for the 1st pdf page
            og.loc[0, "page_number"] = int(og.loc[1, "page_number"]) - 1

            # Extract chapter name from 1st markdown chunk in any case
            match = re.search(
                r".*[a-z]\n", extracted_pdfs["iemh103.pdf"].pages[0].markdown
            )
            og.loc[0, "chapter"] = (
                og.loc[0, "chapter"]
                if og.loc[0, "chapter"] and not is_nan(og.loc[0, "chapter"])
                else match.group(0).strip()
            )

            # Forward fill chapter
            og["page_number"] = og["page_number"].replace("", np.nan).ffill()

            # Forward fill chapter
            og["chapter"] = og["chapter"].replace("", np.nan).ffill()

            # Backward fill subject
            og["subject"] = og["subject"].replace("", np.nan).bfill()

            # 1st page section will always be introduction
            og.at[0, "section"] = (
                og.at[0, "section"]
                if og.at[0, "section"] and not is_nan(og.loc[0, "section"])
                else ["INTRODUCTION"]
            )

            # 1st page description - introduction to "chapter"
            og.loc[0, "page_description"] = (
                og.loc[0, "page_description"]
                if og.loc[0, "page_description"]
                and not is_nan(og.loc[0, "page_description"])
                else f"Introduction to {og.loc[0, "chapter"]}"
            )

            # Section play
            og["section"] = og["section"].apply(ffill_sections)
            update_metadata(extracted_pdf_pages, og.to_dict(orient="records"))
        except Exception:
            if pdf_file_name != "iemh1ps.pdf":
                raise
    return extracted_pdfs


# print(extracted_pages)
# print(extracted_pages['iemh103.pdf'].pages[1])


def process_data():
    extracted_pdfs: dict[str, ExtractedPDF] = {}

    with open(
        "/home/pratik/Documents/Sarvaha/AI-Tutor/data/parsed_pdf_docs_final_2.json",
        "r",
    ) as f:
        data: list[ParsedPDF] = json.load(f)
        extracted_pdfs = create_extracted_pages_object(extracted_pdfs, data)

    extracted_pdfs = extract_metadata(extracted_pdfs)
    extracted_pdfs = post_process_metadata(extracted_pdfs)
    return extracted_pdfs

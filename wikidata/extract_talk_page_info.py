import xml.etree.ElementTree as ET
import csv

def extract_data_from_xml(xml_file):
    try:
        # Load XML data from file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        data = []

        # Iterate over each revision in the XML
        for revision in root.findall('.//revision'):
            if revision is not None:
                # Extract timestamp
                timestamp = revision.find('timestamp').text
                
                # Extract contributor (can be IP or username)
                contributor = revision.find('contributor')
                if contributor.find('username') is not None:
                    user = contributor.find('username').text
                else:
                    user = contributor.find('ip').text

                comment_text = revision.find('comment')
                if comment_text is not None:
                    comment = comment_text.text
                
                # Extract content
                content = revision.find('text').text
                # Remove specified part from content
                if content is not None:
                    content = content.replace('{{Talk header}}', '').replace('{{Outline of knowledge coverage|construction}}', '').replace('{{WikiProjectBannerShell|1=\n{{WikiProject Architecture|class=start|importance=top}}\n{{WikiProject Civil engineering|class=start|importance=}}\n{{WikiProject Technology|class=start|importance=high}}\n}}', '')
                
                # Extract title
                title = root.find('title').text

                # Check if title starts with "Talk:" and edited date is between 2013 - 2023
                if title.startswith("Talk:") and int(timestamp[:4]) < 2024 and int(timestamp[:4]) > 2012:
                    data.append((title, user, timestamp, comment, content))
        
        return data
    
    except ET.ParseError:
        # Handle the case where the XML file is empty or contains no elements
        print(f"Error parsing XML file: {xml_file}")
        return []

def save_to_csv(data, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['title', 'user', 'edited_date', 'edit_comment', 'content'])
        # Write data rows
        writer.writerows(data)

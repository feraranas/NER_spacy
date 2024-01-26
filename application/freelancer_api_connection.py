'''

curl --request GET \
     --url 'https://www.freelancer.com/api/projects/0.1/jobs/search/?job_names%5B%5D=PHP&job_names%5B%5D=website%20design' \
     --header 'freelancer-oauth-v1:K9BPhSaB8rMUOTxb92qy6Xl15IIxPV'

'''

from dotenv import load_dotenv
import requests
import os

load_dotenv()

freelancer_api_key = os.getenv("FREELANCER_API_KEY")

url = 'https://www.freelancer.com/api/projects/0.1/jobs/search/'
headers = {
    'freelancer-oauth-v1': freelancer_api_key
}
params = {
    'job_names[]': ['PHP', 'website design']
}

response = requests.get(url, headers=headers, params=params)

# Print the response
print(response.json())

'''
{'status': 'success', 
 'result': [{'id': 3, 
             'name': 'PHP', 
             'category': {'id': 1, 'name': 'Websites, IT & Software'}, 
             'active_project_count': None, 
             'seo_url': 'php', 
             'seo_info': None, 
             'local': False
             }, 
             {'id': 17, 'name': 'Website Design', 'category': {'id': 3, 'name': 'Design, Media & Architecture'}, 'active_project_count': None, 'seo_url': 'website_design', 'seo_info': None, 'local': False}, 
             {'id': 235, 'name': 'CakePHP', 'category': {'id': 1, 'name': 'Websites, IT & Software'}, 'active_project_count': None, 'seo_url': 'cakephp', 'seo_info': None, 'local': False}, 
             {'id': 385, 'name': 'Symfony PHP', 'category': {'id': 1, 'name': 'Websites, IT & Software'}, 'active_project_count': None, 'seo_url': 'symfony_php', 'seo_info': None, 'local': False}, 
             {'id': 400, 'name': 'Smarty PHP', 'category': {'id': 1, 'name': 'Websites, IT & Software'}, 'active_project_count': None, 'seo_url': 'smarty_php', 'seo_info': None, 'local': False}, 
             {'id': 1020,'name': 'phpMyAdmin', 'category': {'id': 1, 'name': 'Websites, IT & Software'}, 'active_project_count': None, 'seo_url': 'phpmyadmin', 'seo_info': None, 'local': False}, 
             {'id': 1076, 'name': 'PhpNuke', 'category': {'id': 1, 'name': 'Websites, IT & Software'}, 'active_project_count': None, 'seo_url': 'phpnuke', 'seo_info': None, 'local': False}, 
             {'id': 1079, 'name': 'phpFox', 'category': {'id': 1, 'name': 'Websites, IT & Software'}, 'active_project_count': None, 'seo_url': 'phpfox', 'seo_info': None, 'local': False}, 
             {'id': 1526, 'name': 'PHP Slim', 'category': {'id': 1, 'name': 'Websites, IT & Software'}, 'active_project_count': None, 'seo_url': 'php_slim', 'seo_info': None, 'local': False}, 
             {'id': 1904, 'name': 'PHPrunner', 'category': {'id': 1, 'name': 'Websites, IT & Software'}, 'active_project_count': None, 'seo_url': 'phprunner', 'seo_info': None, 'local': False}, 
             {'id': 2401, 'name': 'PHPUnit', 'category': {'id': 1, 'name': 'Websites, IT & Software'}, 'active_project_count': None, 'seo_url': 'phpunit', 'seo_info': None, 'local': False}, 
             {'id': 2627, 'name': 'phpBB', 'category': {'id': 1, 'name': 'Websites, IT & Software'}, 'active_project_count': None, 'seo_url': 'phpbb', 'seo_info': None, 'local': False}, 
             {'id': 2647, 'name': 'Core PHP', 'category': {'id': 1, 'name': 'Websites, IT & Software'}, 'active_project_count': None, 'seo_url': 'core_php', 'seo_info': None, 'local': False}
          ], 
        'request_id': 'e21f54d99dabd6d7dc50a67f0870465f'}
'''
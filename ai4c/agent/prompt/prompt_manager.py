from jinja2 import Environment, FileSystemLoader, select_autoescape


class PromptManager:
    def __init__(self, template_dir: str):
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_name: str, **kwargs) -> str:
        """Render a template with provided context data."""
        template = self.env.get_template(template_name)
        return template.render(**kwargs)

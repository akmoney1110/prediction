from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth import get_user_model

User = get_user_model()

# Optional pretty select for CountryField (django-countries)
try:
    from django_countries.widgets import CountrySelectWidget
    COUNTRY_WIDGET = CountrySelectWidget(attrs={"class": "form-select"})
except Exception:
    COUNTRY_WIDGET = None

class LoginForm(AuthenticationForm):
    remember_me = forms.BooleanField(required=False, initial=False, label="Keep me signed in")

class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        # password1/password2 come from UserCreationForm
        fields = ("username", "email", "country")
        widgets = {"country": COUNTRY_WIDGET} if COUNTRY_WIDGET else {}

    def clean_email(self):
        email = self.cleaned_data["email"].lower()
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("Email already in use.")
        return email



from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import SetPasswordForm
from django.core.validators import EmailValidator

User = get_user_model()

class ForgotPasswordRequestForm(forms.Form):
    email = forms.EmailField(
        validators=[EmailValidator()],
        widget=forms.EmailInput(attrs={
            "class": "w-full rounded border border-white/10 bg-slate-900/60 px-3 py-2 text-sm",
            "placeholder": "you@example.com", "autocomplete": "email",
        })
    )

class ForgotPasswordVerifyForm(forms.Form):
    code = forms.CharField(
        max_length=6, min_length=6,
        widget=forms.TextInput(attrs={
            "class": "w-full rounded border border-white/10 bg-slate-900/60 px-3 py-2 text-sm",
            "placeholder": "6-digit code", "inputmode": "numeric", "autocomplete": "one-time-code",
        })
    )

class ForgotPasswordSetForm(SetPasswordForm):
    # SetPasswordForm expects `user` in __init__
    def __init__(self, user, *args, **kwargs):
        super().__init__(user, *args, **kwargs)
        self.fields["new_password1"].widget.attrs.update({
            "class": "w-full rounded border border-white/10 bg-slate-900/60 px-3 py-2 text-sm",
            "placeholder": "New password", "autocomplete": "new-password",
        })
        self.fields["new_password2"].widget.attrs.update({
            "class": "w-full rounded border border-white/10 bg-slate-900/60 px-3 py-2 text-sm",
            "placeholder": "Confirm new password", "autocomplete": "new-password",
        })



from django import forms
from django.contrib.auth.forms import SetPasswordForm
from django.contrib.auth import get_user_model
from django.core.validators import EmailValidator




# secure/forms.py
from django import forms
from django.contrib.auth import get_user_model

User = get_user_model()

class ForgotPasswordRequestForm(forms.Form):
    email = forms.EmailField(
        validators=[EmailValidator()],
        widget=forms.EmailInput(attrs={
            "class": "w-full rounded border border-white/10 bg-slate-900/60 px-3 py-2 text-sm",
            "placeholder": "you@example.com", "autocomplete": "email",
        })
    )
    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        try:
            user = User.objects.get(email__iexact=email, is_active=True)
        except User.DoesNotExist:
            # ⬇️ Show inline error in the modal
            raise forms.ValidationError("No account found with that email.")
        # stash the user for the view
        self.user = user
        return email






class ForgotPasswordVerifyForm(forms.Form):
    code = forms.CharField(
        max_length=6, min_length=6,
        widget=forms.TextInput(attrs={
            "class": "w-full rounded border border-white/10 bg-slate-900/60 px-3 py-2 text-sm",
            "placeholder": "6-digit code", "inputmode": "numeric", "autocomplete": "one-time-code",
        })
    )

class ForgotPasswordSetForm(SetPasswordForm):
    def __init__(self, user, *args, **kwargs):
        super().__init__(user, *args, **kwargs)
        self.fields["new_password1"].widget.attrs.update({
            "class": "w-full rounded border border-white/10 bg-slate-900/60 px-3 py-2 text-sm",
            "placeholder": "New password", "autocomplete": "new-password",
        })
        self.fields["new_password2"].widget.attrs.update({
            "class": "w-full rounded border border-white/10 bg-slate-900/60 px-3 py-2 text-sm",
            "placeholder": "Confirm new password", "autocomplete": "new-password",
        })

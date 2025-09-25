from django.shortcuts import render

# Create your views here.
from django.urls import reverse_lazy
from django.views.generic import CreateView
from django.contrib.auth.views import LoginView
from django.contrib.auth import login
from .forms import SignUpForm, LoginForm

# secure/views.py
from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.auth import login
from django.urls import reverse_lazy
from django.views.generic import CreateView
from .forms import SignUpForm

from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.views.generic import CreateView
from django.contrib.auth import login
from django.contrib import messages
from django.utils.http import url_has_allowed_host_and_scheme

from .forms import SignUpForm

# secure/views.py
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.views.generic import CreateView
from django.contrib.auth import login
from django.contrib import messages
from django.utils.http import url_has_allowed_host_and_scheme
from .forms import SignUpForm
# views.py
from django.contrib import messages
from django.contrib.auth import login
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.generic import CreateView

from .forms import SignUpForm  # adjust import if needed


class SignUpView(CreateView):
    form_class = SignUpForm
    template_name = "signup.html"  # full-page fallback

    def _wants_fragment(self):
        return self.request.headers.get("HX-Request") or self.request.GET.get("fragment")

    def get_template_names(self):
        return ["_signup_form.html"] if self._wants_fragment() else [self.template_name]

    def get_success_url(self):
        # Prefer a safe ?next=... if provided
        next_url = self.request.GET.get("next") or self.request.POST.get("next")
        if next_url and url_has_allowed_host_and_scheme(
            next_url, allowed_hosts={self.request.get_host()}
        ):
            return next_url
        # Fallback to your dashboard with required kwargs
        return reverse("combined_dashboard", kwargs={"league_id": 0, "days": 14})

    def form_valid(self, form):
        self.object = form.save()
        login(self.request, self.object)
        if self._wants_fragment():
            resp = render(self.request, "_signup_success.html", {}) if False else None  # optional
            # Redirect the modal via HTMX
            from django.http import HttpResponse
            resp = HttpResponse("")  # empty body; HTMX will follow redirect
            resp["HX-Redirect"] = self.get_success_url()
            return resp
        return redirect(self.get_success_url())

    def form_invalid(self, form):
        if self._wants_fragment():
            resp = render(self.request, "_signup_form.html", {"form": form})
            resp["HX-Retarget"] = "#signup-modal-body"
            resp["HX-Reswap"] = "innerHTML"
            return resp

        messages.error(self.request, "Please fix the errors in the form and try again.")
        referer = self.request.META.get("HTTP_REFERER")
        if referer and url_has_allowed_host_and_scheme(referer, {self.request.get_host()}):
            return redirect(referer)
        return self.render_to_response(self.get_context_data(form=form))



from django.contrib.auth.views import LoginView
from django.http import HttpResponseRedirect
from django.conf import settings

# secure/views.py
from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.auth.views import LoginView
from .forms import LoginForm

class CustomLoginView(LoginView):
    authentication_form = LoginForm
    template_name = "login.html"  # full-page fallback

    def _wants_fragment(self):
        return (
            self.request.headers.get("HX-Request")
            or self.request.GET.get("fragment")
        )

    def get_template_names(self):
        if self._wants_fragment():
            return ["_login_form.html"]     # modal fragment
        return [self.template_name]

    def form_valid(self, form):
        # handle "remember me"
        remember = form.cleaned_data.get("remember_me")
        self.request.session.set_expiry(60 * 60 * 24 * 30 if remember else 0)

        # Let LoginView log the user in
        response = super().form_valid(form)

        # If this was an HTMX submit, redirect via HX header (keeps us in modal context)
        if self._wants_fragment():
            resp = HttpResponse("")
            resp["HX-Redirect"] = self.get_success_url()
            return resp
        return response

    def form_invalid(self, form):
        # push a single friendly error
        messages.error(self.request, "Invalid username/email or password.", extra_tags="login")
        # prefer going back to where the user submitted from
        referer = self.request.META.get("HTTP_REFERER")
        if referer:
            return redirect(referer)
        # fallback: login page (preserve next if present)
        messages.error(self.request, "Invalid username/email or password.")
        next_url = self.request.POST.get("next") or self.request.GET.get("next")
        if next_url:
            return redirect(f"{reverse('secure:login')}?next={next_url}")
        return redirect("secure:login")

import secrets, datetime
from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import make_password, check_password
from django.shortcuts import render, redirect
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_http_methods
from django.contrib import messages

from .forms import ForgotPasswordRequestForm, ForgotPasswordVerifyForm, ForgotPasswordSetForm
from .models import PasswordResetCode
from .emailing import send_brevo_email

User = get_user_model()

RESET_SESSION_USER_KEY = "pw_reset_user_id"
RESET_SESSION_OK_KEY   = "pw_reset_ok_until"

def _isue_code_for_user(user: User) -> None:
    # basic rate-limit: 1 code per minute
    one_min_ago = timezone.now() - datetime.timedelta(minutes=1)
    if PasswordResetCode.objects.filter(user=user, created_at__gte=one_min_ago, consumed_at__isnull=True).exists():
        return

    code = f"{secrets.randbelow(1_000_000):06d}"
    expires = timezone.now() + datetime.timedelta(minutes=10)
    PasswordResetCode.objects.create(user=user, code_hash=make_password(code), expires_at=expires)

    html = f"""
      <p>Hello {user.username},</p>
      <p>Your password reset code is:</p>
      <p style="font-size:20px;"><strong>{code}</strong></p>
      <p>This code expires in 10 minutes. If you didn’t request this, you can ignore this email.</p>
    """
    send_brevo_email(user.email, "Your password reset code", html, sender_name="Surfwithak")

@require_http_methods(["GET", "POST"])
def fogot_password_request(request):
    if request.method == "POST":
        form = ForgotPasswordRequestForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["email"].lower()
            # Do not reveal if the email exists
            user = User.objects.filter(email__iexact=email, is_active=True).first()
            if user:
                request.session[RESET_SESSION_USER_KEY] = user.id
                _issue_code_for_user(user)
            messages.success(request, "If that email exists, we sent a code.")
            return redirect("secure:forgot_verify")
    else:
        form = ForgotPasswordRequestForm()
    return render(request, "forgot_request.html", {"form": form})

@require_http_methods(["GET", "POST"])
def fogot_password_verify(request):
    user_id = request.session.get(RESET_SESSION_USER_KEY)
    if not user_id:
        return redirect("secure:forgot")

    user = User.objects.filter(id=user_id, is_active=True).first()
    if not user:
        return redirect("secure:forgot")

    form = ForgotPasswordVerifyForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        code = form.cleaned_data["code"].strip()
        code_obj = (PasswordResetCode.objects
                    .filter(user=user, consumed_at__isnull=True, expires_at__gt=timezone.now())
                    .order_by("-created_at").first())
        if not code_obj:
            form.add_error("code", "Code expired. Request a new one.")
        else:
            if code_obj.attempt_count >= 5:
                form.add_error("code", "Too many attempts. Request a new code.")
            else:
                code_obj.attempt_count += 1
                code_obj.save(update_fields=["attempt_count"])
                if check_password(code, code_obj.code_hash):
                    code_obj.consumed_at = timezone.now()
                    code_obj.save(update_fields=["consumed_at"])
                    # allow set step for 5 minutes
                    request.session[RESET_SESSION_OK_KEY] = (timezone.now() + datetime.timedelta(minutes=5)).isoformat()
                    return redirect("secure:forgot_new")
                else:
                    form.add_error("code", "Invalid code.")
    return render(request, "forgot_verify.html", {"form": form, "email": user.email})

@require_http_methods(["GET", "POST"])
def forgt_password_set(request):
    user_id = request.session.get(RESET_SESSION_USER_KEY)
    ok_until_iso = request.session.get(RESET_SESSION_OK_KEY)
    if not user_id or not ok_until_iso:
        return redirect("secure:forgot")

    try:
        ok_until = timezone.datetime.fromisoformat(ok_until_iso)
        if timezone.is_naive(ok_until):
            ok_until = timezone.make_aware(ok_until, timezone.get_current_timezone())
    except Exception:
        return redirect("secure:forgot")

    if timezone.now() > ok_until:
        messages.error(request, "Session expired. Request a new code.")
        return redirect("secure:forgot")

    user = User.objects.filter(id=user_id, is_active=True).first()
    if not user:
        return redirect("secure:forgot")

    form = ForgotPasswordSetForm(user, request.POST or None)
    if request.method == "POST" and form.is_valid():
        form.save()  # sets new password
        # Invalidate any other outstanding codes for safety
        PasswordResetCode.objects.filter(user=user, consumed_at__isnull=True).update(consumed_at=timezone.now())
        # clear session keys
        request.session.pop(RESET_SESSION_USER_KEY, None)
        request.session.pop(RESET_SESSION_OK_KEY, None)
        messages.success(request, "Password reset. You can sign in now.")
        return redirect("secure:login")

    return render(request, "forgot_set.html", {"form": form})










# secure/views.py (only the forgot* views shown)
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.utils import timezone
from django.contrib import messages
from django.urls import reverse
import datetime
from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import make_password, check_password

from .forms import ForgotPasswordRequestForm, ForgotPasswordVerifyForm, ForgotPasswordSetForm
from .models import PasswordResetCode

User = get_user_model()
RESET_SESSION_USER_KEY = "pw_reset_user_id"
RESET_SESSION_OK_KEY   = "pw_reset_ok_until"

def _wants_fragment(request):
    return request.headers.get("HX-Request") or request.GET.get("fragment")

def _issue_code_for_user(user: User):
    one_min_ago = timezone.now() - datetime.timedelta(minutes=1)
    if PasswordResetCode.objects.filter(user=user, created_at__gte=one_min_ago, consumed_at__isnull=True).exists():
        return
    import secrets
    code = f"{secrets.randbelow(1_000_000):06d}"
    expires = timezone.now() + datetime.timedelta(minutes=10)
    PasswordResetCode.objects.create(user=user, code_hash=make_password(code), expires_at=expires)

    # ✉️ send via your Brevo helper
    from .emailing import send_brevo_email
    html = f"""
      <p>Hello {user.username},</p>
      <p>Your password reset code is:</p>
      <p style="font-size:20px;"><strong>{code}</strong></p>
      <p>This code expires in 10 minutes. If you didn’t request this, ignore this email.</p>
    """
    send_brevo_email(user.email, "Your password reset code", html, sender_name="Surfwithak")



@require_http_methods(["GET", "POST"])
def forgot_password_request(request):
    if request.method == "POST":
        form = ForgotPasswordRequestForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["email"].lower()
            user = User.objects.filter(email__iexact=email, is_active=True).first()
            if user:
                request.session[RESET_SESSION_USER_KEY] = user.id
                _issue_code_for_user(user)
            # Do not reveal existence of the email
            if _wants_fragment(request):
                # Move to step 2 in the modal
                return render(request, "_forgot_verify.html", {
                    "form": ForgotPasswordVerifyForm(),
                    "email": email,
                })
            messages.success(request, "If that email exists, we sent a code.")
            return redirect("secure:forgot_verify")
        # invalid email format etc.
        if _wants_fragment(request):
            return render(request, "_forgot_request.html", {"form": form})
    else:
        form = ForgotPasswordRequestForm()

    if _wants_fragment(request):
        return render(request, "_forgot_request.html", {"form": form})
    return render(request, "secure/forgot_request.html", {"form": form})  # full-page fallback



@require_http_methods(["GET", "POST"])
def forgot_password_verify(request):
    user_id = request.session.get(RESET_SESSION_USER_KEY)
    user = User.objects.filter(id=user_id, is_active=True).first() if user_id else None
    if not user:
        if _wants_fragment(request):
            return render(request, "_forgot_request.html", {"form": ForgotPasswordRequestForm()})
        return redirect("secure:forgot")

    if request.method == "POST":
        form = ForgotPasswordVerifyForm(request.POST)
        if form.is_valid():
            code = form.cleaned_data["code"].strip()
            code_obj = (PasswordResetCode.objects
                        .filter(user=user, consumed_at__isnull=True, expires_at__gt=timezone.now())
                        .order_by("-created_at").first())
            if code_obj and code_obj.attempt_count < 5 and check_password(code, code_obj.code_hash):
                code_obj.consumed_at = timezone.now()
                code_obj.save(update_fields=["consumed_at"])
                # allow set step for 5 minutes
                request.session[RESET_SESSION_OK_KEY] = (timezone.now() + datetime.timedelta(minutes=5)).isoformat()

                if _wants_fragment(request):
                    # Move to step 3 in the modal
                    return render(request, "_forgot_set.html", {"form": ForgotPasswordSetForm(user)})
                return redirect("secure:forgot_new")
            # invalid or expired
            if code_obj:
                code_obj.attempt_count += 1
                code_obj.save(update_fields=["attempt_count"])
            form.add_error("code", "Invalid or expired code.")
            if _wants_fragment(request):
                return render(request, "_forgot_verify.html", {"form": form, "email": user.email})

    # GET
    if _wants_fragment(request):
        return render(request, "_forgot_verify.html", {"form": ForgotPasswordVerifyForm(), "email": user.email})
    return render(request, "forgot_verify.html", {"form": ForgotPasswordVerifyForm(), "email": user.email})



@require_http_methods(["GET", "POST"])
def forgot_password_set(request):
    user_id = request.session.get(RESET_SESSION_USER_KEY)
    ok_until_iso = request.session.get(RESET_SESSION_OK_KEY)
    user = User.objects.filter(id=user_id, is_active=True).first() if user_id else None
    if not user or not ok_until_iso:
        if _wants_fragment(request):
            return render(request, "_forgot_request.html", {"form": ForgotPasswordRequestForm()})
        return redirect("secure:forgot")

    # check window
    try:
        from django.utils import timezone as djtz
        ok_until = djtz.datetime.fromisoformat(ok_until_iso)
        if djtz.is_naive(ok_until):
            ok_until = djtz.make_aware(ok_until, djtz.get_current_timezone())
        if djtz.now() > ok_until:
            if _wants_fragment(request):
                return render(request, "_forgot_request.html", {"form": ForgotPasswordRequestForm()})
            return redirect("secure:forgot")
    except Exception:
        if _wants_fragment(request):
            return render(request, "_forgot_request.html", {"form": ForgotPasswordRequestForm()})
        return redirect("secure:forgot")

    if request.method == "POST":
        form = ForgotPasswordSetForm(user, request.POST)
        if form.is_valid():
            form.save()
            # invalidate leftovers and clear session
            PasswordResetCode.objects.filter(user=user, consumed_at__isnull=True).update(consumed_at=timezone.now())
            request.session.pop(RESET_SESSION_USER_KEY, None)
            request.session.pop(RESET_SESSION_OK_KEY, None)

            if _wants_fragment(request):
                # Show success inside the modal (no page nav)
                return render(request, "_forgot_success.html", {})
            return redirect("secure:login")

        if _wants_fragment(request):
            return render(request, "_forgot_set.html", {"form": form})

    # GET
    if _wants_fragment(request):
        return render(request, "_forgot_set.html", {"form": ForgotPasswordSetForm(user)})
    return render(request, "forgot_set.html", {"form": ForgotPasswordSetForm(user)})


# views.py
from django.contrib.auth import logout
from django.shortcuts import redirect
from django.contrib import messages

def custom_logout(request):
    logout(request)
    messages.success(request, "You have been successfully logged out.")
    return redirect('combined_dashboard', league_id=0, days=14)   # Redirect to your home page or login page
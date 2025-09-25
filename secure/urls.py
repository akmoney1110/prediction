from django.urls import path
from .views import SignUpView, CustomLoginView,custom_logout,forgot_password_request,forgot_password_verify,forgot_password_set
from django.contrib.auth.views import LogoutView, PasswordResetView, PasswordResetDoneView, PasswordResetConfirmView, PasswordResetCompleteView

urlpatterns = [
    path("signup/", SignUpView.as_view(), name="signup"),
    path("login/", CustomLoginView.as_view(), name="login"),

    path('logout/', custom_logout, name='logout'),

    path("password_reset/",          PasswordResetView.as_view(),         name="password_reset"),
    path("password_reset/done/",     PasswordResetDoneView.as_view(),     name="password_reset_done"),
    path("reset/<uidb64>/<token>/",  PasswordResetConfirmView.as_view(),  name="password_reset_confirm"),
    path("reset/done/",              PasswordResetCompleteView.as_view(), name="password_reset_complete"),
    path("forgot/",        forgot_password_request, name="forgot"),
    path("forgot/verify/", forgot_password_verify,  name="forgot_verify"),
    path("forgot/new/",    forgot_password_set,     name="forgot_new"),
]

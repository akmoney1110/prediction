from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion

class Migration(migrations.Migration):

    dependencies = [
        ('matches', '0001_initial'),  # <- change if your last matches migration isn’t 0001
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='UserDailyTicket',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(default='My ticket', blank=True, max_length=120)),
                ('filters', models.JSONField(default=dict, blank=True)),
                ('ticket_date', models.DateField(db_index=True)),
                ('selections', models.JSONField(default=list)),
                ('legs', models.IntegerField(default=0)),
                ('acc_probability', models.FloatField(blank=True, null=True)),
                ('acc_fair_odds', models.FloatField(blank=True, null=True)),
                ('acc_bookish_odds', models.FloatField(blank=True, null=True)),
                ('status', models.CharField(choices=[('pending', 'Pending'), ('won', 'Won'), ('lost', 'Lost'), ('void', 'Void')], default='pending', max_length=12)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('base_ticket', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='user_variants', to='matches.dailyticket')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='daily_tickets', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-ticket_date', '-created_at'],
            },
        ),
        migrations.AddIndex(
            model_name='userdailyticket',
            index=models.Index(fields=['user', 'ticket_date'], name='matches_use_user_id_efd4f8_idx'),
        ),
    ]

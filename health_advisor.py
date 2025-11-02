"""
Health Advisory Engine
Provides personalized health recommendations based on AQI and user demographics
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthAdvisor:
    """
    Generates health advisories and recommendations based on AQI levels
    Tailored for vulnerable groups (elderly, children, asthmatics, etc.)
    """
    
    def __init__(self):
        """Initialize health advisor"""
        self.general_advisories = self._load_general_advisories()
        self.vulnerable_group_advisories = self._load_vulnerable_advisories()
    
    def _load_general_advisories(self) -> Dict:
        """Load general health advisories for each AQI category"""
        return {
            'Good': {
                'summary': 'Air quality is satisfactory',
                'health_impact': 'Air quality is considered satisfactory, and air pollution poses little or no risk.',
                'recommendations': [
                    'Perfect day for outdoor activities',
                    'All groups can enjoy normal outdoor activities',
                    'Windows can be kept open for ventilation'
                ],
                'outdoor_activity': 'Recommended',
                'mask_needed': False,
                'color': config.AQI_CATEGORIES['Good']['color']
            },
            'Satisfactory': {
                'summary': 'Air quality is acceptable',
                'health_impact': 'Air quality is acceptable for most people. However, sensitive people may experience minor breathing discomfort.',
                'recommendations': [
                    'Enjoy outdoor activities',
                    'Sensitive individuals should watch for symptoms',
                    'Consider closing windows during peak traffic hours'
                ],
                'outdoor_activity': 'Recommended',
                'mask_needed': False,
                'color': config.AQI_CATEGORIES['Satisfactory']['color']
            },
            'Moderate': {
                'summary': 'May cause breathing discomfort to sensitive people',
                'health_impact': 'May cause breathing discomfort to people with lung disease such as asthma, and discomfort to people with heart disease, children, and older adults.',
                'recommendations': [
                    'Limit prolonged outdoor exertion',
                    'Sensitive groups should reduce outdoor activities',
                    'Keep windows closed during high pollution hours',
                    'Consider using air purifiers indoors'
                ],
                'outdoor_activity': 'Moderate with caution',
                'mask_needed': True,
                'mask_type': 'N95 or N99 recommended for sensitive groups',
                'color': config.AQI_CATEGORIES['Moderate']['color']
            },
            'Poor': {
                'summary': 'May cause breathing discomfort to most people',
                'health_impact': 'May cause breathing discomfort to most people on prolonged exposure, and respiratory illness on long exposure.',
                'recommendations': [
                    'Avoid prolonged outdoor activities',
                    'Wear N95/N99 masks when outdoors',
                    'Keep doors and windows closed',
                    'Use air purifiers with HEPA filters',
                    'Avoid outdoor exercise',
                    'Stay hydrated'
                ],
                'outdoor_activity': 'Not recommended',
                'mask_needed': True,
                'mask_type': 'N95 or N99 mandatory',
                'color': config.AQI_CATEGORIES['Poor']['color']
            },
            'Very Poor': {
                'summary': 'May cause respiratory effects even on healthy people',
                'health_impact': 'May cause respiratory effects even on healthy people, and serious health impacts on people with lung/heart diseases. The health impacts may be experienced even during light physical activity.',
                'recommendations': [
                    'Stay indoors as much as possible',
                    'Avoid all outdoor physical activities',
                    'Wear N99 masks if you must go outside',
                    'Use air purifiers continuously',
                    'Keep emergency medications handy',
                    'Monitor health symptoms closely',
                    'Avoid travel if possible'
                ],
                'outdoor_activity': 'Strongly discouraged',
                'mask_needed': True,
                'mask_type': 'N99 mandatory, consider respirators',
                'color': config.AQI_CATEGORIES['Very Poor']['color']
            },
            'Severe': {
                'summary': 'Emergency conditions - health warnings',
                'health_impact': 'Severe health effects on all population groups. Everyone may experience more serious health effects.',
                'recommendations': [
                    '🚨 HEALTH EMERGENCY - Stay indoors',
                    'Do not go outside unless absolutely necessary',
                    'Wear N99 masks or respirators if you must go out',
                    'Run air purifiers on high',
                    'Seal windows and doors',
                    'Avoid all physical exertion',
                    'Keep emergency medications accessible',
                    'Monitor news for official advisories',
                    'Consider relocating temporarily if possible'
                ],
                'outdoor_activity': 'Prohibited',
                'mask_needed': True,
                'mask_type': 'N99 or full respirator mandatory',
                'emergency': True,
                'color': config.AQI_CATEGORIES['Severe']['color']
            }
        }
    
    def _load_vulnerable_advisories(self) -> Dict:
        """Load specific advisories for vulnerable groups"""
        return {
            'elderly': {
                'name': 'Elderly (65+ years)',
                'icon': '👵👴',
                'risk_factors': [
                    'Reduced lung capacity',
                    'Weaker immune system',
                    'Existing health conditions'
                ],
                'threshold_modifier': -50,  # More sensitive, lower threshold
                'specific_advice': {
                    'Moderate': [
                        'Stay indoors during peak pollution hours (6-10 AM, 6-10 PM)',
                        'Take prescribed medications regularly',
                        'Keep emergency contacts handy'
                    ],
                    'Poor': [
                        'Avoid going outdoors',
                        'Monitor blood pressure and heart rate',
                        'Consult doctor if experiencing discomfort'
                    ],
                    'Very Poor': [
                        'Complete indoor stay recommended',
                        'Have caregiver assistance available',
                        'Keep medical emergency kit ready'
                    ]
                }
            },
            'children': {
                'name': 'Children (0-12 years)',
                'icon': '👶👧👦',
                'risk_factors': [
                    'Developing respiratory system',
                    'Higher breathing rate',
                    'More time spent outdoors'
                ],
                'threshold_modifier': -50,
                'specific_advice': {
                    'Moderate': [
                        'Limit outdoor playtime',
                        'Schools should avoid outdoor sports',
                        'Ensure children drink plenty of water'
                    ],
                    'Poor': [
                        'No outdoor activities',
                        'Keep schools closed or indoor-only',
                        'Watch for coughing or breathing difficulty'
                    ],
                    'Very Poor': [
                        'Keep children indoors at all times',
                        'Cancel school if possible',
                        'Monitor for respiratory symptoms closely'
                    ]
                }
            },
            'asthmatic': {
                'name': 'Asthma Patients',
                'icon': '🫁',
                'risk_factors': [
                    'Sensitive airways',
                    'Risk of asthma attacks',
                    'Increased medication needs'
                ],
                'threshold_modifier': -75,
                'specific_advice': {
                    'Satisfactory': [
                        'Keep rescue inhaler accessible',
                        'Monitor for early symptoms'
                    ],
                    'Moderate': [
                        'Use preventive inhaler before going out',
                        'Avoid outdoor exercise',
                        'Wear N95 mask when outdoors'
                    ],
                    'Poor': [
                        'Stay indoors',
                        'Increase preventer medication as advised by doctor',
                        'Keep emergency inhaler within reach',
                        'Have emergency action plan ready'
                    ],
                    'Very Poor': [
                        'Complete indoor isolation',
                        'Be prepared to visit hospital if needed',
                        'Monitor peak flow regularly',
                        'Avoid all triggers including smoke, dust'
                    ]
                }
            },
            'heart_disease': {
                'name': 'Heart Disease Patients',
                'icon': '❤️',
                'risk_factors': [
                    'Increased cardiovascular stress',
                    'Higher risk of heart attacks',
                    'Blood pressure fluctuations'
                ],
                'threshold_modifier': -60,
                'specific_advice': {
                    'Moderate': [
                        'Avoid physical exertion',
                        'Monitor blood pressure regularly',
                        'Take medications as prescribed'
                    ],
                    'Poor': [
                        'Stay indoors and rest',
                        'Check blood pressure twice daily',
                        'Keep emergency medications ready',
                        'Have someone check on you regularly'
                    ],
                    'Very Poor': [
                        'Complete rest recommended',
                        'Constant monitoring advised',
                        'Have emergency contact on speed dial',
                        'Consider hospitalization if symptoms worsen'
                    ]
                }
            },
            'pregnant': {
                'name': 'Pregnant Women',
                'icon': '🤰',
                'risk_factors': [
                    'Affects fetal development',
                    'Increased respiratory needs',
                    'Higher vulnerability to pollution'
                ],
                'threshold_modifier': -65,
                'specific_advice': {
                    'Moderate': [
                        'Limit time outdoors',
                        'Wear N95 mask when going out',
                        'Stay well-hydrated',
                        'Rest frequently'
                    ],
                    'Poor': [
                        'Stay indoors',
                        'Use air purifiers',
                        'Attend only essential medical appointments',
                        'Monitor fetal movements'
                    ],
                    'Very Poor': [
                        'Complete indoor isolation',
                        'Postpone non-emergency medical visits',
                        'Consult doctor via telemedicine',
                        'Watch for any unusual symptoms'
                    ]
                }
            }
        }
    
    def get_advisory(self, 
                    aqi_value: float,
                    vulnerable_groups: Optional[List[str]] = None) -> Dict:
        """
        Get health advisory for given AQI value
        
        Args:
            aqi_value: Current or forecasted AQI
            vulnerable_groups: List of vulnerable group identifiers
            
        Returns:
            Comprehensive health advisory dict
        """
        # Determine AQI category
        category_info = config.get_aqi_category(aqi_value)
        category = category_info['category']
        
        # Get general advisory
        general = self.general_advisories.get(category, {})
        
        advisory = {
            'aqi_value': round(aqi_value, 1),
            'category': category,
            'color': category_info['color'],
            'icon': category_info['icon'],
            'summary': general.get('summary', ''),
            'health_impact': general.get('health_impact', ''),
            'recommendations': general.get('recommendations', []),
            'outdoor_activity': general.get('outdoor_activity', ''),
            'mask_needed': general.get('mask_needed', False),
            'mask_type': general.get('mask_type', 'Not required'),
            'emergency': general.get('emergency', False),
            'generated_at': datetime.now().isoformat()
        }
        
        # Add vulnerable group specific advisories
        if vulnerable_groups:
            advisory['vulnerable_groups'] = {}
            
            for group in vulnerable_groups:
                if group in self.vulnerable_group_advisories:
                    group_info = self.vulnerable_group_advisories[group]
                    
                    # Adjust category for vulnerable groups
                    adjusted_aqi = aqi_value - group_info['threshold_modifier']
                    adjusted_category_info = config.get_aqi_category(adjusted_aqi)
                    adjusted_category = adjusted_category_info['category']
                    
                    advisory['vulnerable_groups'][group] = {
                        'name': group_info['name'],
                        'icon': group_info['icon'],
                        'risk_level': adjusted_category,
                        'risk_factors': group_info['risk_factors'],
                        'specific_advice': group_info['specific_advice'].get(
                            adjusted_category, []
                        )
                    }
        
        return advisory
    
    def get_forecast_alert(self,
                          current_aqi: float,
                          forecast_aqi: List[float],
                          location: str,
                          vulnerable_groups: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Generate alert if forecast shows deteriorating conditions
        
        Args:
            current_aqi: Current AQI value
            forecast_aqi: List of forecasted AQI values (next 24 hours)
            location: Location name
            vulnerable_groups: User's vulnerable group memberships
            
        Returns:
            Alert dict if conditions warrant, None otherwise
        """
        max_forecast = max(forecast_aqi)
        avg_forecast = sum(forecast_aqi) / len(forecast_aqi)
        
        # Check if conditions are deteriorating significantly
        if max_forecast > config.ALERT_THRESHOLD_AQI:
            
            # Find when peak will occur
            peak_hour = forecast_aqi.index(max_forecast) + 1
            peak_time = datetime.now().replace(minute=0, second=0) + \
                       timedelta(hours=peak_hour)
            
            category_info = config.get_aqi_category(max_forecast)
            
            alert = {
                'type': 'forecast_deterioration',
                'severity': 'high' if max_forecast > 300 else 'medium',
                'location': location,
                'current_aqi': round(current_aqi, 1),
                'peak_aqi': round(max_forecast, 1),
                'average_aqi': round(avg_forecast, 1),
                'peak_category': category_info['category'],
                'peak_time': peak_time.strftime('%I:%M %p'),
                'peak_hour': peak_hour,
                'message': f"⚠️ Air quality alert for {location}! "
                          f"AQI expected to reach {int(max_forecast)} "
                          f"({category_info['category']}) around {peak_time.strftime('%I:%M %p')}.",
                'recommendations': self.get_advisory(
                    max_forecast, vulnerable_groups
                )['recommendations'],
                'generated_at': datetime.now().isoformat()
            }
            
            return alert
        
        return None
    
    def get_comparison_advice(self,
                             current_aqi: float,
                             forecast_aqi: float) -> str:
        """
        Get comparative advice (improving/worsening)
        
        Args:
            current_aqi: Current AQI
            forecast_aqi: Forecasted AQI
            
        Returns:
            Advisory message
        """
        diff = forecast_aqi - current_aqi
        pct_change = (diff / current_aqi) * 100
        
        if abs(diff) < 10:
            return "Air quality expected to remain similar. Maintain current precautions."
        elif diff > 0:
            if pct_change > 25:
                return f"⚠️ Air quality expected to worsen significantly ({int(pct_change)}% increase). Plan accordingly and limit outdoor exposure."
            else:
                return f"Air quality expected to deteriorate slightly. Consider reducing outdoor activities."
        else:
            if abs(pct_change) > 25:
                return f"✅ Good news! Air quality expected to improve significantly ({int(abs(pct_change))}% decrease)."
            else:
                return f"Air quality expected to improve slightly. Still maintain precautions."
    
    def get_whatsapp_message(self,
                            aqi_value: float,
                            location: str,
                            url: str) -> str:
        """
        Generate WhatsApp-friendly message
        
        Args:
            aqi_value: Current/forecasted AQI
            location: Location name
            url: Dashboard URL
            
        Returns:
            Formatted message for sharing
        """
        category_info = config.get_aqi_category(aqi_value)
        
        message = f"""🌍 *Delhi Air Quality Alert*

📍 Location: {location}
🔢 AQI: {int(aqi_value)}
📊 Category: {category_info['category']} {category_info['icon']}

{self.general_advisories[category_info['category']]['summary']}

*Key Recommendations:*
"""
        
        recs = self.general_advisories[category_info['category']]['recommendations'][:3]
        for rec in recs:
            message += f"• {rec}\n"
        
        message += f"\n🔗 View live dashboard: {url}\n"
        message += f"\n⏰ Updated: {datetime.now().strftime('%I:%M %p, %d %b %Y')}"
        
        return message


if __name__ == "__main__":
    # Example usage
    advisor = HealthAdvisor()
    
    # Get advisory for moderate pollution
    print("\n1. General Advisory (AQI 150):")
    advisory = advisor.get_advisory(150)
    print(f"   Category: {advisory['category']}")
    print(f"   Summary: {advisory['summary']}")
    print(f"   Mask needed: {advisory['mask_needed']}")
    
    # Get advisory for vulnerable groups
    print("\n2. Advisory for Asthma Patient (AQI 150):")
    advisory = advisor.get_advisory(150, vulnerable_groups=['asthmatic'])
    if 'asthmatic' in advisory['vulnerable_groups']:
        print(f"   Risk Level: {advisory['vulnerable_groups']['asthmatic']['risk_level']}")
        print(f"   Advice: {advisory['vulnerable_groups']['asthmatic']['specific_advice']}")
    
    # Get forecast alert
    print("\n3. Forecast Alert:")
    current = 120
    forecast = [125, 135, 150, 180, 220, 250] + [200] * 18
    alert = advisor.get_forecast_alert(current, forecast, "Anand Vihar")
    if alert:
        print(f"   {alert['message']}")
    
    # Get WhatsApp message
    print("\n4. WhatsApp Message:")
    msg = advisor.get_whatsapp_message(180, "Connaught Place", "https://delhiaqi.app")
    print(msg)
